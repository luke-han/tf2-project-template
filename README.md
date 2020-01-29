# Tensorflow 2 Project Template
Inspired from https://github.com/Ahmkel/Keras-Project-Template

[Pix2Pix](https://phillipi.github.io/pix2pix/) is implemented as a sample model.

## Features
- supports multi-gpu training
- supports save model and load model in order to continue to train the model
- backups source code used to train

## How to train
### Requirements
python >= 3.7

### Install required packages
```shell
pip install -r requirements.txt
```

### Download pix2pix datasets
```bash
./datasets/download_pix2pix_dataset.sh facades
```

### Setup the config file
edit configs/pix2pix.yml file

### Train
```shell
python train.py --config configs/pix2pix.yml
```

### Train from a certain checkpoint
```shell
python train.py --config configs/pix2pix.yml --checkpoint 100 
```

### Run Tensorboard
```shell
tensorboard --logdir /path/to/experiment/dir/tensorboard/
```