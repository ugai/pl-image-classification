# pl-image-classification

Image classification example for PyTorch Lightning.

## Requirements

- Python >= 3.10

## Setup

Clone this repository.

```sh
git clone https://github.com/ugai/pl-image-classification
```

Create a local environment.

```sh
cd pl-image-classification
python -m venv env
./env/Scripts/activate
pip install -r requirements.txt
```

## Training

Activate the local envrionment.

```sh
cd pl-image-classification
./env/Scripts/activate
```

Run the training process.

```sh
python tutorial.py train <DATASET_ROOT_DIR> --num-classes <NUM_OF_CLASSES>
```

## Inference

Activate the local envrionment.

```sh
cd pl-image-classification
./env/Scripts/activate
```

Run the inference process.

```sh
python tutorial.py infer <CKPT_FILE> <IMAGE_DIR> --num-classes <NUM_OF_CLASSES>
```
