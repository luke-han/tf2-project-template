import os
from functools import partial
from glob import glob

import tensorflow as tf
from dotmap import DotMap

from base.data_loader import DataLoader
from utils.image import resize_image, load_image, normalize_image


@tf.function
def vertical_split(image):
    w = tf.shape(image)[1] // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]
    return input_image, real_image


@tf.function
def random_crop(input_image, real_image, height, width):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, height, width, 3])

    return cropped_image[0], cropped_image[1]


@tf.function
def random_jitter(input_image, real_image, height, width, resize_ratio_before_crop):
    # resize images
    resize_height = int(resize_ratio_before_crop * height)
    resize_width = int(resize_ratio_before_crop * width)
    input_image = resize_image(input_image, resize_height, resize_width)
    real_image = resize_image(real_image, resize_height, resize_width)

    # randomly cropping
    input_image, real_image = random_crop(input_image, real_image, height, width)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


@tf.function
def load_image_train(image_file, height, width, resize_ratio_before_crop):
    input_image, real_image = vertical_split(load_image(image_file))
    input_image, real_image = random_jitter(input_image, real_image, height, width, resize_ratio_before_crop)
    input_image, real_image = normalize_image(input_image), normalize_image(real_image)

    return input_image, real_image


@tf.function
def load_image_test(image_file, height, width):
    input_image, real_image = vertical_split(load_image(image_file))
    input_image, real_image = resize_image(input_image, height, width), resize_image(real_image, height, width)
    input_image, real_image = normalize_image(input_image), normalize_image(real_image)

    return input_image, real_image


class Pix2PixDataLoader(DataLoader):
    def __init__(self, config: DotMap, strategy: tf.distribute.Strategy) -> None:
        super().__init__(config, strategy)
        self.base_path = f'./datasets/{self.config.dataset.name}/'

    def get_train_dataset(self) -> tf.data.Dataset:
        map_func = partial(load_image_train,
                           height=self.config.dataset.image_size,
                           width=self.config.dataset.image_size,
                           resize_ratio_before_crop=self.config.dataset.data_loader.resize_ratio_before_crop)
        train_dataset = tf.data.Dataset.list_files(os.path.join(self.base_path, 'train/*.jpg'))
        train_dataset = train_dataset.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.config.dataset.data_loader.shuffle_buffer_size)
        train_dataset = train_dataset.batch(self.global_batch_size)
        return self.strategy.experimental_distribute_dataset(train_dataset)

    def get_validation_dataset(self) -> tf.data.Dataset:
        map_func = partial(load_image_test,
                           height=self.config.dataset.image_size,
                           width=self.config.dataset.image_size)
        val_dataset = tf.data.Dataset.list_files(os.path.join(self.base_path, 'val/*.jpg'))
        val_dataset = val_dataset.map(map_func)
        val_dataset = val_dataset.batch(self.global_batch_size)
        return self.strategy.experimental_distribute_dataset(val_dataset)

    def get_test_dataset(self) -> tf.data.Dataset:
        map_func = partial(load_image_test,
                           height=self.config.dataset.image_size,
                           width=self.config.dataset.image_size)
        test_dataset = tf.data.Dataset.list_files(os.path.join(self.base_path, 'test/*.jpg'))
        test_dataset = test_dataset.map(map_func)
        test_dataset = test_dataset.batch(self.global_batch_size)
        return self.strategy.experimental_distribute_dataset(test_dataset)

    def get_train_data_size(self) -> int:
        return len(glob(os.path.join(self.base_path, 'train/*.jpg')))

    def get_validation_data_size(self) -> int:
        return len(glob(os.path.join(self.base_path, 'val/*.jpg')))

    def get_test_data_size(self) -> int:
        return len(glob(os.path.join(self.base_path, 'test/*.jpg')))
