import tensorflow as tf


def patch_gan_discriminator():
    kernel_initializer = tf.random_normal_initializer(0., 0.02)
    kernel_size = 4

    input_image = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    target_image = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

    x = tf.keras.layers.Concatenate()([input_image, target_image])

    for filters, batch_norm in [(64, False), (128, True), (256, True)]:
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same',
                                   kernel_initializer=kernel_initializer, use_bias=False)(x)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=kernel_size, strides=1, kernel_initializer=kernel_initializer,
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(1, kernel_size=kernel_size, strides=1, kernel_initializer=kernel_initializer)(x)

    return tf.keras.Model(inputs=[input_image, target_image], outputs=x)
