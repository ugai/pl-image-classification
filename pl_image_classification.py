import os
import shutil
import time
from datetime import datetime
from hashlib import blake2b
from multiprocessing import cpu_count, freeze_support
from pathlib import Path
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import typer
from PIL import Image
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler
from rich.progress import track
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.models._api import Weights
from torchvision.transforms._presets import ImageClassification

from backbones import BackboneType, get_backbone_model_and_weights
from visualization import preview_dataset_images, preview_image

app = typer.Typer()

Image.init()  # init Image.EXTENSION list
Image.MAX_IMAGE_PIXELS = None  # disable PIL.Image.DecompressionBombWarning


class Defaults:
    """Default arguments"""

    NUM_CLASSES: int = 2
    ACCELERATOR: str = "auto"
    BACKBONE_TYPE: BackboneType = BackboneType.ResNet50


class CachedImageFolder(ImageFolder):
    """Cached ImageFolder"""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        base_cache_dir: Path = Path(".cache", "imgs"),
        resize_size: tuple[int] = (224,),
    ):
        # Replace root to cache_root
        self.base_cache_dir = base_cache_dir
        self.resize_size = resize_size
        self.create_cache_dir(root)
        root = str(self.cache_root)

        super().__init__(root, transform, target_transform, loader, is_valid_file)
        print("CachedImageFolder initialized")

    def create_cache_dir(self, src_root: str):
        src_imgfolder = ImageFolder(src_root)
        class_count = len(src_imgfolder.classes)
        img_count = len(src_imgfolder.imgs)
        hash_str = f"c{class_count}_n{img_count}_{src_imgfolder.root}"
        dataset_hash = blake2b(hash_str.encode(), digest_size=8).hexdigest()

        self.cache_root = self.base_cache_dir / str(dataset_hash)

        if self.cache_root.exists():
            sub_dir_count = sum(1 for _ in self.cache_root.iterdir())
            if sub_dir_count == class_count:
                file_count = 0
                for sub_dir in self.cache_root.iterdir():
                    file_count += sum(1 for _ in sub_dir.glob("*"))
                if file_count == img_count:
                    return  # has cache

            # remove previous cache
            shutil.rmtree(self.cache_root)

        os.makedirs(self.cache_root)
        for class_name in src_imgfolder.classes:
            class_dir = self.cache_root / class_name
            os.makedirs(class_dir)

        for src_path, class_idx in track(
            src_imgfolder.imgs, description="Processing..."
        ):
            class_name = src_imgfolder.classes[class_idx]
            file_name = os.path.basename(src_path)
            dst_path = self.cache_root / class_name / file_name
            img = Image.open(src_path)
            img = img.resize((self.resize_size[0], self.resize_size[0]))
            img.save(dst_path)


class SubsetDataset(Dataset):
    """Treat a Subset as Dataset"""

    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y


class SingleFolderDataset(Dataset):
    def __init__(self, src_dir: Path, transform=None):
        self.src_dir = src_dir
        self.transform = transform

        extensions = tuple(Image.EXTENSION.keys())

        self.image_paths = []
        self.labels = []
        for entry_name in os.listdir(src_dir):
            path = os.path.join(src_dir, entry_name)
            if entry_name.lower().endswith(extensions) and os.path.isfile(path):
                self.image_paths.append(path)
                self.labels.append(0)  # currently not used

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        y_dummy = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return x, y_dummy


# LightningDataModule
# https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningDataModule.html#lightningdatamodule
class AutoSplitImageFolderDataModule(pl.LightningDataModule):
    """My Lightning Data Module"""

    def __init__(
        self,
        weights: Weights,
        image_folder: Path,
        use_cache: bool,
        batch_size: int = 1,
    ):
        super().__init__()

        # Save Hyperparameters
        # https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#hyperparameters-in-datamodules
        self.save_hyperparameters(ignore=["weights"])

        self.weights = weights
        self.image_folder = image_folder
        self.use_cache = use_cache
        # Define the self.batch_size to enable batch size finder
        self.batch_size = batch_size
        self.num_workers = cpu_count()
        self.pin_memory = True  # for speed
        self.persistent_workers = True
        self.train_ratio = 0.7

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        self.dataset = (
            CachedImageFolder(str(self.image_folder))
            if self.use_cache
            else ImageFolder(str(self.image_folder))
        )

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        indices = list(range(len(self.dataset.targets)))

        # Split dataset
        #   train = train_ratio
        #   val   = (1 - train_ratio) / 2
        #   test  = (1 - train_ratio) / 2
        train_indices, other_indices = train_test_split(
            indices, train_size=self.train_ratio, stratify=self.dataset.targets
        )
        other_dataset_targets = [self.dataset.targets[v] for v in other_indices]
        val_indices, test_indices = train_test_split(
            other_indices, test_size=0.5, stratify=other_dataset_targets
        )

        # Get transforms from pretained model weights
        preprocess: ImageClassification = self.weights.transforms()

        train_transforms = nn.Sequential(
            transforms.Resize(
                size=preprocess.resize_size,
                interpolation=preprocess.interpolation,
            ),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomPerspective(),
            transforms.AugMix(),
            # transforms.TrivialAugmentWide(),
            preprocess,
        )

        def create_sub_dataset(indices, transforms):
            return SubsetDataset(Subset(self.dataset, indices), transforms)

        self.train_dataset = create_sub_dataset(train_indices, train_transforms)
        self.val_dataset = create_sub_dataset(val_indices, preprocess)
        self.test_dataset = create_sub_dataset(test_indices, preprocess)

        # preview_dataset_images(self.train_dataset, wait_time=5)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass


# LightningModule
# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#inference-in-production
class ClassificationTask(pl.LightningModule):
    """My Lightning Module"""

    def __init__(self, model, num_classes: int, learning_rate: float = 0.001):
        super().__init__()

        # Save Hyperparameters
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters(ignore=["model"])

        all_modules = list(model.children())
        last_module = all_modules[-1]

        if isinstance(last_module, nn.Sequential):
            # Get existing feature-extraction layers
            self.feature_extractor = nn.Sequential(
                *all_modules[:-1],
                last_module[:-1],
            )
            # Create new classification layer
            self.classifier = nn.Linear(last_module[-1].in_features, num_classes)  # type: ignore
        else:
            # Get existing feature-extraction layers
            self.feature_extractor = nn.Sequential(*all_modules[:-1])
            # Create new classification layer
            self.classifier = nn.Linear(last_module.in_features, num_classes)
        self.model = nn.Sequential(self.feature_extractor, self.classifier)

        # Loss function
        self.criterion = F.cross_entropy

        # Optimizer
        # Define the self.learning_rate to enable learning rate finder
        # https://pytorch-lightning.readthedocs.io/en/1.4.5/advanced/lr_finder.html
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.RAdam(
            self.classifier.parameters(), lr=self.learning_rate
        )

        # Create a MetricsCollection to use multiple metrics
        # https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metriccollection
        task = "multiclass"
        base_coll = MetricCollection(
            Accuracy(task=task, num_classes=num_classes, average="macro"),
            Precision(task=task, num_classes=num_classes, average="macro"),
            Recall(task=task, num_classes=num_classes, average="macro"),
            F1Score(task=task, num_classes=num_classes, average="macro"),
            AUROC(task=task, num_classes=num_classes, average="macro"),
            MatthewsCorrCoef(task=task, num_classes=num_classes),
        )

        # Clone the base MetricsCollection to train/val/test
        self.metrics_colls = {}
        self._metrics_colls = nn.ModuleList()
        for loop_name in ["train", "val", "test"]:
            self.metrics_colls[loop_name] = base_coll.clone(prefix=f"{loop_name}_")
            # All Metrics should be accessible as class members that inherit from nn.Module
            # https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metrics-and-devices
            self._metrics_colls.append(self.metrics_colls[loop_name])
        del base_coll

    def forward(self, x):  # pylint: disable=arguments-differ
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch):  # pylint: disable=arguments-differ
        loss = self._shared_eval_step("train", batch)
        return loss

    def training_epoch_end(self, outputs):
        losses = [output["loss"] for output in outputs]  # type: ignore
        self._shared_eval_epoch_end("train", losses)

    def validation_step(self, batch, _batch_idx):  # pylint: disable=arguments-differ
        loss = self._shared_eval_step("val", batch)
        return loss

    def validation_epoch_end(self, outputs):
        losses = outputs
        self._shared_eval_epoch_end("val", losses)

    def test_step(self, batch, _batch_idx):  # pylint: disable=arguments-differ
        loss = self._shared_eval_step("test", batch)
        return loss

    def test_epoch_end(self, outputs):
        losses = outputs
        self._shared_eval_epoch_end("test", losses)

    def _shared_eval_step(self, loop_name: str, batch):
        x, y = batch
        y_hat = self(x)  # forward
        loss = self.criterion(y_hat, y)
        self.metrics_colls[loop_name].update(y_hat, y)
        return loss

    def _shared_eval_epoch_end(self, loop_name: str, outputs):
        metrics = self.metrics_colls[loop_name].compute()
        metrics[f"{loop_name}_loss"] = torch.stack(outputs).mean()  # type: ignore
        self.log_dict(metrics, prog_bar=True)
        self.metrics_colls[loop_name].reset()

    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        x, _y = batch
        y_hat = self(x)  # forward
        return y_hat

    def configure_optimizers(self):
        return self.optimizer


class FileBasedPauseTrainCallback(pl.Callback):
    def __init__(self, watch_file_path: str = ".pausetrain", wait_time: int = 10):
        self.watch_file_path = watch_file_path
        self.wait_time = wait_time
        print(
            f"FileBasedPauseTrainCallback: watch start '{self.watch_file_path}' at on_train_epoch_end"
        )

    def on_train_epoch_end(self, _trainer, _pl_module):
        if self.has_file():
            print(f"FileBasedPauseTrainCallback: pause start, {datetime.now()}")
            while self.has_file():
                time.sleep(self.wait_time)
            print(f"FileBasedPauseTrainCallback: pause end, {datetime.now()}")

    def has_file(self) -> bool:
        return os.path.isfile(self.watch_file_path)


@app.command()
def train(
    image_folder: Path,
    use_cache: bool = True,
    backbone_type: BackboneType = Defaults.BACKBONE_TYPE,
    num_classes: int = Defaults.NUM_CLASSES,
    max_epoch: int = 10,
    check_val_every_n_epoch: int = 5,
    batch_size: int = 256,
    early_stopping: bool = True,
    auto_lr_find: bool = True,
    auto_scale_batch_size: bool = True,
    accelerator: str = Defaults.ACCELERATOR,
    use_profiler: bool = False,
    log_save_dir: str = "",
):  # pylint: disable=too-many-arguments
    """Training task"""
    backbone_model, backbone_weights = get_backbone_model_and_weights(backbone_type)

    model = ClassificationTask(model=backbone_model, num_classes=num_classes)

    logger = TensorBoardLogger(log_save_dir, name=f"lightning_logs/{backbone_type}")
    profiler = SimpleProfiler(filename="perf_logs") if use_profiler else None
    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            monitor="val_MulticlassMatthewsCorrCoef",  # Monitor log() or log_dict() key
            mode="max",
        ),
        StochasticWeightAveraging(swa_lrs=1e-2),
        FileBasedPauseTrainCallback(),
    ]
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_MulticlassMatthewsCorrCoef",  # Monitor log() or log_dict() key
                mode="max",
                patience=10,  # patience_epochs = check_val_every_n_epoch * patience,
            )
        )
    trainer = pl.Trainer(
        max_epochs=max_epoch,
        accelerator=accelerator,
        precision="bf16",
        check_val_every_n_epoch=min(check_val_every_n_epoch, max_epoch),
        log_every_n_steps=5,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=auto_scale_batch_size,
        auto_select_gpus=True,
        benchmark=True,
        logger=logger,
        profiler=profiler,
        callbacks=callbacks,
    )

    data_module = AutoSplitImageFolderDataModule(
        backbone_weights,
        image_folder,
        use_cache,
        batch_size,
    )

    # Run a learning rate finder and a batch size finder
    trainer.tune(model, datamodule=data_module)

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")


@app.command()
def infer(
    model_path: Path,
    image_folder: Path,
    backbone_type: BackboneType = Defaults.BACKBONE_TYPE,
    num_classes: int = Defaults.NUM_CLASSES,
    accelerator: str = Defaults.ACCELERATOR,
    preview: bool = False,
    preview_timeout: float = 5.0,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Prediction task"""
    backbone_model, backbone_weights = get_backbone_model_and_weights(backbone_type)
    model = ClassificationTask.load_from_checkpoint(
        checkpoint_path=model_path, model=backbone_model, num_classes=num_classes
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        precision="bf16",
        auto_select_gpus=True,
        callbacks=RichProgressBar(),  # type: ignore
        inference_mode=True,
    )

    dataset = SingleFolderDataset(image_folder, transform=backbone_weights.transforms())
    data_loader = DataLoader(dataset)
    predictions = trainer.predict(model, data_loader)
    predictions = [(int(torch.argmax(pred)), pred) for pred in predictions]  # type: ignore

    for i, (class_idx, tensor) in enumerate(predictions):
        image_path = dataset.image_paths[i]
        print(f"{i}: {class_idx}, {list(tensor)}, '{image_path}'")

        if preview:
            preview_image(image_path, f"class={class_idx}", preview_timeout)


if __name__ == "__main__":
    freeze_support()
    app()
