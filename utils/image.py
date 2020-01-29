import tensorflow as tf


@tf.function
def load_image(image_file: str) -> tf.Tensor:
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    return tf.cast(image, tf.float32)


@tf.function
def resize_image(input_image: tf.Tensor, height: int, width: int) -> tf.Tensor:
    return tf.image.resize(input_image, [height, width])


# normalizing the image to [-1, 1]
@tf.function
def normalize_image(input_image: tf.Tensor) -> tf.Tensor:
    return (input_image / 127.5) - 1


@tf.function
def denormalize_image(input_image: tf.Tensor) -> tf.Tensor:
    return input_image * 127.5 + 127.5
