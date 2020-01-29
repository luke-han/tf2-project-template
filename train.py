import argparse
import os
import shutil
import socket

import tensorflow as tf
from dotmap import DotMap

from base.data_loader import DataLoader
from data_loader.pix2pix_data_loader import Pix2PixDataLoader
from model_trainer_builder import build_model_and_get_trainer
from utils.config import process_config


def get_data_loader(config: DotMap, strategy: tf.distribute.Strategy) -> DataLoader:
    data_loader_type = config.dataset.data_loader.type
    with strategy.scope():
        if data_loader_type == 'pix2pix':
            return Pix2PixDataLoader(config, strategy)
        else:
            raise ValueError(f"unknown data loader type {data_loader_type}")


def backup_source_codes(config: DotMap) -> None:
    if not os.path.exists(config.exp.source_dir):
        # copy source files
        shutil.copytree(
            os.path.abspath(os.path.curdir),
            config.exp.source_dir,
            ignore=lambda src, names: {"datasets", "__pycache__", ".git", "experiments", "venv"})


def main(config_path: str, checkpoint: int) -> None:
    config = process_config(config_path, checkpoint)

    backup_source_codes(config)

    strategy = tf.distribute.MirroredStrategy()
    print('number of devices: {}'.format(strategy.num_replicas_in_sync))

    data_loader = get_data_loader(config=config, strategy=strategy)

    trainer = build_model_and_get_trainer(config, data_loader, strategy)

    print(f"Start Training Experiment {config.exp.name}")
    trainer.train()


if __name__ == '__main__':
    print(socket.gethostname(), os.path.abspath(os.curdir), os.getpid())
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/pix2pix.yml", help="config path to use")
    ap.add_argument("--checkpoint", type=int, default=0, help="checkpoint to continue")

    args = vars(ap.parse_args())
    main(args["config"], args["checkpoint"])
