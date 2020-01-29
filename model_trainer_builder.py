import tensorflow as tf
from dotmap import DotMap

from base.data_loader import DataLoader
from base.trainer import Trainer
from models.patchgan_discriminator import patch_gan_discriminator
from models.unet_generator import unet_generator
from trainers.pix2pix_trainer import Pix2PixTrainer


def get_generator_model(config: DotMap) -> tf.keras.Model:
    model_name = config.model.generator.model
    if model_name == 'unet':
        return unet_generator()
    else:
        raise ValueError(f"unknown generator model {model_name}")


def get_discriminator_model(config: DotMap) -> tf.keras.Model:
    model_name = config.model.discriminator.model
    if model_name == 'patchgan':
        return patch_gan_discriminator()
    else:
        raise ValueError(f"unknown discriminator model {model_name}")


# returns combined_model (for load saved model), trainer
def build_model_and_get_trainer(config: DotMap, data_loader: DataLoader, strategy: tf.distribute.Strategy) -> Trainer:
    model_structure = config.model.structure

    print('Create the model')
    if model_structure == 'pix2pix':
        with strategy.scope():
            generator = get_generator_model(config)
            discriminator = get_discriminator_model(config)

        trainer = Pix2PixTrainer(generator=generator, discriminator=discriminator, data_loader=data_loader,
                                 strategy=strategy, config=config)

        return trainer
    else:
        raise ValueError(f"unknown model structure {model_structure}")
