import datetime
import os
from collections import defaultdict
from typing import Iterable, Union

import tensorflow as tf
from PIL import Image
from dotmap import DotMap

from base.data_loader import DataLoader
from base.trainer import Trainer
from utils.image import denormalize_image


class Pix2PixTrainer(Trainer):
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model,
                 data_loader: DataLoader, strategy: tf.distribute.Strategy, config: DotMap) -> None:
        super().__init__(data_loader, strategy, config)
        self.generator: tf.keras.Model = generator
        self.discriminator: tf.keras.Model = discriminator

        with strategy.scope():
            self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.model.generator.lr,
                                                                beta_1=config.model.generator.beta1)

            self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.model.discriminator.lr,
                                                                    beta_1=config.model.discriminator.beta1)
            self.disc_real_accuracy = tf.keras.metrics.BinaryAccuracy(name='real_accuracy')
            self.disc_fake_accuracy = tf.keras.metrics.BinaryAccuracy(name='fake_accuracy')

        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(config.exp.tensorboard_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=generator,
                                              discriminator=discriminator)

    @tf.function
    def train_step(self, inputs):
        gen_adversarial_weight = self.config.model.generator.adversarial_weight
        gen_l1_weight = self.config.model.generator.l1_weight
        global_batch_size = self.config.trainer.batch_size * self.strategy.num_replicas_in_sync

        @tf.function
        def generator_loss(valid_fake, fake, target):
            # gan loss
            bce_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                 reduction=tf.keras.losses.Reduction.NONE)
            gan_loss = bce_loss_object(tf.ones_like(valid_fake), valid_fake)
            gan_loss = tf.reduce_mean(gan_loss, axis=[-2, -1])
            gan_loss = tf.nn.compute_average_loss(gan_loss, global_batch_size=global_batch_size)

            # mean absolute error
            mae_loss_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
            l1_loss = mae_loss_object(fake, target)
            l1_loss = tf.reduce_mean(l1_loss, axis=[-2, -1])
            l1_loss = tf.nn.compute_average_loss(l1_loss, global_batch_size=global_batch_size)

            total_gen_loss = gen_adversarial_weight * gan_loss + gen_l1_weight * l1_loss

            return total_gen_loss, gan_loss, l1_loss

        @tf.function
        def discriminator_loss(valid_real, valid_fake):
            loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

            # real
            real_loss = loss_object(tf.ones_like(valid_real), valid_real)
            real_loss = tf.reduce_mean(real_loss, axis=[-2, -1])
            real_loss = tf.nn.compute_average_loss(real_loss, global_batch_size=global_batch_size)

            # fake
            fake_loss = loss_object(tf.zeros_like(valid_fake), valid_fake)
            fake_loss = tf.reduce_mean(fake_loss, axis=[-2, -1])
            fake_loss = tf.nn.compute_average_loss(fake_loss, global_batch_size=global_batch_size)

            total_disc_loss = real_loss + fake_loss
            return total_disc_loss, real_loss, fake_loss

        input_image, target_image = inputs
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target_image], training=True)
            disc_fake_output = self.discriminator([input_image, gen_output], training=True)

            gen_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_fake_output, gen_output, target_image)
            disc_loss, disc_real_loss, disc_fake_loss = discriminator_loss(disc_real_output, disc_fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        self.disc_real_accuracy.update_state(tf.ones_like(disc_real_output), disc_real_output)
        self.disc_fake_accuracy.update_state(tf.zeros_like(disc_fake_output), disc_fake_output)

        return [gen_loss, gen_gan_loss, gen_l1_loss], [disc_loss, disc_real_loss, disc_fake_loss]

    @tf.function
    def distributed_train_step(self, inputs):
        gen_losses, disc_losses = self.strategy.experimental_run_v2(self.train_step, args=(inputs,))
        gen_losses = [self.strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=None) for x in gen_losses]
        disc_losses = [self.strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=None) for x in disc_losses]
        return gen_losses, disc_losses

    @staticmethod
    def g_metric_names():
        return ['loss/G', 'loss/G_adv', 'loss/G_L1']

    @staticmethod
    def d_metric_names():
        return ['loss/D', 'loss/D_real', 'loss/D_fake']

    def train(self):
        epochs = self.config.trainer.num_epochs

        batch_size = self.config.trainer.batch_size
        steps_per_epoch = self.data_loader.get_train_data_size() // batch_size // self.strategy.num_replicas_in_sync
        assert steps_per_epoch > 0

        train_dataset = self.data_loader.get_train_dataset()
        valid_dataset = self.data_loader.get_validation_dataset()

        if self.config.trainer.epoch_to_continue > 0:
            with self.strategy.scope():
                checkpoint_dir = f"{self.config.exp.checkpoints_dir}/{self.config.trainer.epoch_to_continue:04d}/"
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                status = self.checkpoint.restore(latest_checkpoint)
                status.assert_existing_objects_matched()
                print(f"restored from checkpoint {latest_checkpoint}")

        self.on_train_begin()
        with self.summary_writer.as_default():
            for epoch in range(self.config.trainer.epoch_to_continue, epochs):
                epoch_logs = defaultdict(float)
                for step, inputs in enumerate(train_dataset):
                    gen_losses, disc_losses = self.distributed_train_step(inputs)

                    if epoch == self.config.trainer.epoch_to_continue and step == 0:    # check only once
                        assert len(gen_losses) == len(self.g_metric_names())
                        assert len(disc_losses) == len(self.d_metric_names())

                    metric_logs = {}
                    for name, value in zip(self.g_metric_names() + self.d_metric_names(), gen_losses + disc_losses):
                        metric_logs[name] = value.numpy()
                    metric_logs['accuracy/D_real'] = self.disc_real_accuracy.result() * 100
                    metric_logs['accuracy/D_fake'] = self.disc_fake_accuracy.result() * 100
                    self.disc_real_accuracy.reset_states()
                    self.disc_fake_accuracy.reset_states()

                    for name, value in metric_logs.items():
                        epoch_logs[name] += value / steps_per_epoch

                    print(f"[Epoch {epoch + 1}/{epochs}] [Batch {step + 1}/{steps_per_epoch}]")
                    print(',\t'.join([f"{name}={value:.1f}%" if 'accuracy' in name else f"{name}={value:.4f}" if 'loss' in name else f"{name}={value}" for name, value in metric_logs.items()]), flush=True)

                    if step + 1 == steps_per_epoch and (epoch + 1) % self.config.trainer.predict_freq == 0:
                        self.sample_images([inputs], f"{self.config.exp.sample_dir}/train/", epoch + 1)

                # additional logs
                epoch_logs['lr/G'] = self.generator_optimizer.lr.numpy()
                epoch_logs['lr/D'] = self.discriminator_optimizer.lr.numpy()

                for key, value in epoch_logs.items():
                    tf.summary.scalar(key, value, epoch)

                if (epoch + 1) % self.config.trainer.predict_freq == 0:
                    self.sample_images(valid_dataset, f"{self.config.exp.sample_dir}/valid/{epoch + 1:04d}", epoch + 1)

                if (epoch + 1) % self.config.trainer.checkpoint_freq == 0 and (epoch + 1 != epochs):
                    prefix = f"{self.config.exp.checkpoints_dir}/{epoch + 1:04d}/"
                    os.makedirs(prefix, exist_ok=True)
                    self.checkpoint.save(file_prefix=prefix)

            self.save_trained_model(epochs)
            test_dataset = self.data_loader.get_test_dataset()
            self.sample_images(test_dataset, f"{self.config.exp.sample_dir}/test/{epochs:04d}", epochs)

    def save_trained_model(self, epochs):
        prefix = f"{self.config.exp.saved_models_dir}/{epochs:04d}/"
        os.makedirs(prefix, exist_ok=True)
        self.generator.save(f"{prefix}/generator")
        self.discriminator.save(f"{prefix}/discriminator")

    def sample_images(self, dataset: Union[tf.data.Dataset, Iterable], output_dir: str, epoch: int) -> None:
        os.makedirs(output_dir, exist_ok=True)

        images = []
        for inputs in dataset:
            input_image, target_image = inputs
            # The training=True is intentional here
            # since we want the batch statistics while running the model on the test dataset.
            # If we use training=False, we will get the accumulated statistics learned
            # from the training dataset (which we don't want)
            gen_output = self.generator(input_image, training=True)

            for i in range(gen_output.shape[0]):
                image = tf.concat([input_image[i], target_image[i], gen_output[i]], axis=1)
                images.append(denormalize_image(image))

        save_batch_size = self.config.trainer.batch_size
        for i in range(0, len(images), save_batch_size):
            concat_images = tf.concat(images[i:i + save_batch_size], axis=0)
            output_name = f"{output_dir}/{epoch:04d}_{i // save_batch_size}.png"
            Image.fromarray(tf.cast(concat_images, tf.uint8).numpy()).save(output_name)
