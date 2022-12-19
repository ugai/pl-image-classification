import math

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset


def preview_dataset_images(
    dataset: Dataset,
    ncols: int = 10,
    nrows: int = 5,
    wait_time: float | None = None,
    image_save_path: str | None = None,
):
    nsamples = ncols * nrows
    nrows = math.ceil(nsamples / ncols)

    figsize_inches = (ncols, nrows)
    plt.figure(figsize=figsize_inches)

    for i in range(nsamples):
        image, _label = dataset[i]
        image = image.permute(1, 2, 0)
        plt.subplot(nrows, ncols, 1 + i)
        plt.axis("off")
        plt.imshow(image)

    if image_save_path:
        plt.savefig(image_save_path)

    if wait_time is not None:
        plt.waitforbuttonpress(wait_time)


def preview_image(image_path: str, title: str, wait_time: float | None):
    image = Image.open(image_path)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image)  # type: ignore

    if wait_time is not None:
        plt.waitforbuttonpress(wait_time)
