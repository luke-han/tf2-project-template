import abc

import tensorflow as tf
from dotmap import DotMap


class DataLoader:
    def __init__(self, config: DotMap, strategy: tf.distribute.Strategy) -> None:
        super().__init__()
        self.config = config
        self.strategy = strategy
        # Each training batch the dataset creates will be split up onto each GPU
        self.global_batch_size = self.config.trainer.batch_size * self.strategy.num_replicas_in_sync

    @abc.abstractmethod
    def get_train_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError

    @abc.abstractmethod
    def get_validation_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError

    @abc.abstractmethod
    def get_train_data_size(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_validation_data_size(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_data_size(self) -> int:
        raise NotImplementedError
