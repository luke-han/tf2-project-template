import tensorflow as tf


def unet_generator():
    kernel_initializer = tf.random_normal_initializer(0., 0.02)
    kernel_size = 4

    input_image = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    down_stack = []
    # down-sample layers
    for filters, batch_norm in [(64, False), (128, True), (256, True), (512, True), (512, True), (512, True),
                                (512, True), (512, True)]:
        block = tf.keras.Sequential()
        block.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same',
                                         kernel_initializer=kernel_initializer, use_bias=False))
        if batch_norm:
            block.add(tf.keras.layers.BatchNormalization())
        block.add(tf.keras.layers.LeakyReLU())
        down_stack.append(block)

    # up-sample
    up_stack = []
    for filters, dropout in [(512, True), (512, True), (512, True), (512, False), (256, False), (128, False),
                             (64, False)]:
        block = tf.keras.Sequential()
        block.add(
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding='same',
                                            kernel_initializer=kernel_initializer, use_bias=False))
        if dropout:
            block.add(tf.keras.layers.Dropout(rate=0.5))
        block.add(tf.keras.layers.ReLU())
        up_stack.append(block)

    last_layer = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=kernel_size, strides=2, padding='same',
                                                 kernel_initializer=kernel_initializer, activation='tanh',
                                                 name='fake_output')

    x = input_image

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last_layer(x)

    return tf.keras.Model(inputs=input_image, outputs=x)
