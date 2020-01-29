import os
from datetime import datetime

import yaml
from dotmap import DotMap


def get_config_from_yml(yml_file: str) -> DotMap:
    # parse the configurations from the config json file provided
    with open(yml_file, "r") as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config


def process_config(yml_file: str, epoch_to_continue: int) -> DotMap:
    config = get_config_from_yml(yml_file)

    # set checkpoint to continue
    config.trainer.epoch_to_continue = epoch_to_continue

    # set experiment info and create directories to save infos
    exp_dir = os.path.join(config.exp.experiment_dir, config.exp.name)

    config.exp.tensorboard_dir = os.path.join(exp_dir, "tensorboard")
    os.makedirs(config.exp.tensorboard_dir, exist_ok=True)

    config.exp.sample_dir = os.path.join(exp_dir, "samples")
    os.makedirs(config.exp.sample_dir, exist_ok=True)

    config.exp.checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(config.exp.checkpoints_dir, exist_ok=True)

    config.exp.saved_models_dir = os.path.join(exp_dir, "saved_models")
    os.makedirs(config.exp.saved_models_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    config.exp.source_dir = os.path.join(exp_dir, f"source/{current_time}")

    return config
