import abc

import tensorflow as tf
from dotmap import DotMap

from base.data_loader import DataLoader


class Trainer(tf.keras.callbacks.Callback):
    def __init__(self, data_loader: DataLoader, strategy: tf.distribute.Strategy, config: DotMap) -> None:
        self.data_loader: DataLoader = data_loader
        self.strategy = strategy
        self.config: DotMap = config

    @abc.abstractmethod
    def train(self) -> None:
        raise NotImplementedError
