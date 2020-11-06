
import numpy as np
import time
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
sys.path.append("..")
from utils.data_utils import generate_and_save_images


# ****************** network definition  *****************

def make_generator_model(args):
    latent_input = keras.Input(shape=(args.latent_size,))

    both = layers.Dense(8 * 8 * 100, activation='linear', use_bias=False)(latent_input)
    both = layers.Reshape((8, 8, 100))(both)
    both = layers.BatchNormalization()(both)
    both = layers.Activation("relu")(both)
    
    pooled = layers.UpSampling2D(size=(2, 2))(both)
    pooled = layers.Conv2D(64, 5, activation='relu', padding='same')(pooled)
    pooled = layers.Conv2D(64, 5, activation='relu', padding='same')(pooled)

    pooled = layers.UpSampling2D(size=(2, 2))(pooled)
    pooled = layers.Conv2D(32, 5, activation='relu', padding='same')(pooled)
    pooled = layers.Conv2D(32, 5, activation='linear', padding='same')(pooled)

    strided = layers.Conv2DTranspose(64, 5, strides=(2, 2), activation='relu', padding='same')(both)
    strided = layers.Conv2DTranspose(32, 5, strides=(2, 2), activation='linear', padding='same')(strided)

    both = layers.concatenate([pooled, strided], axis=3)  # 32 x 32
    both = layers.BatchNormalization()(both)
    both = layers.Activation("relu")(both)

    pooled = layers.UpSampling2D(size=(2, 2))(both)
    pooled = layers.Conv2D(32, 3, activation='relu', padding='same')(pooled)
    pooled = layers.Conv2D(32, 3, activation='relu', padding='same')(pooled)

    pooled = layers.UpSampling2D(size=(2, 2))(pooled)
    pooled = layers.Conv2D(32, 3, activation='relu', padding='same')(pooled)
    pooled = layers.Conv2D(32, 3, activation='linear', padding='same')(pooled)

    strided = layers.Conv2DTranspose(32, 7, strides=(2, 2), activation='relu', padding='same')(both)
    strided = layers.Conv2DTranspose(32, 7, strides=(2, 2), activation='linear', padding='same')(strided)

    both = layers.concatenate([pooled, strided], axis=3)  # 128 x 128
    both = layers.Cropping2D(cropping=((1, 2), (1, 2)))(both)  # 125 x 125
    both = layers.BatchNormalization()(both)
    both = layers.Activation("relu")(both)

    pooled = layers.UpSampling2D(size=(2, 2))(both)
    pooled = layers.Conv2D(16, 3, activation='relu', padding='same')(pooled)
    pooled = layers.Conv2D(16, 3, activation='relu', padding='same')(pooled)

    pooled = layers.UpSampling2D(size=(2, 2))(pooled)
    pooled = layers.Conv2D(8, 3, activation='relu', padding='same')(pooled)
    pooled = layers.Conv2D(8, 3, activation='linear', padding='same')(pooled)

    strided = layers.Conv2DTranspose(16, 7, strides=(2, 2), activation='relu', padding='same')(both)
    strided = layers.Conv2DTranspose(8, 7, strides=(2, 2), activation='linear', padding='same')(strided)

    both = layers.concatenate([pooled, strided], axis=3)  # 500 x 500
    both = layers.BatchNormalization()(both)
    both = layers.Activation("relu")(both)

    out = layers.Conv2DTranspose(1, 9, strides=(2, 2), activation='sigmoid', padding='same')(both)  # 1000 x 1000

    model = keras.Model(inputs=latent_input, outputs=out, name='generator_model')
    assert model.output_shape == (None, args.img_size, args.img_size, 1)

    return model


def make_inference_model(args):
    img_input = keras.Input(shape=(args.img_size, args.img_size, 1))

    pre = layers.Conv2D(8, 3, activation='relu', padding='same')(img_input)
    pre = layers.Conv2D(8, 3, activation='relu', padding='same')(pre)
    pre = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pre)

    pooled = layers.Conv2D(16, 3, activation='relu', padding='same')(pre)
    pooled = layers.Conv2D(16, 3, activation='relu', padding='same')(pooled)
    pooled = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pooled)

    pooled = layers.Conv2D(32, 3, activation='relu', padding='same')(pooled)
    pooled = layers.Conv2D(32, 3, activation='relu', padding='same')(pooled)
    pooled = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pooled)

    strided = layers.Conv2D(16, 7, strides=(2, 2), activation='relu', padding='same')(pre)
    strided = layers.Conv2D(32, 7, strides=(2, 2), activation='relu', padding='same')(strided)

    both = layers.concatenate([pooled, strided], axis=3)

    pooled = layers.Conv2D(32, 3, activation='relu', padding='same')(both)
    pooled = layers.Conv2D(32, 3, activation='relu', padding='same')(pooled)
    pooled = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pooled)

    pooled = layers.Conv2D(64, 3, activation='relu', padding='same')(pooled)
    pooled = layers.Conv2D(64, 3, activation='relu', padding='same')(pooled)
    pooled = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pooled)

    strided = layers.Conv2D(32, 7, strides=(2, 2), activation='relu', padding='same')(both)
    strided = layers.Conv2D(64, 7, strides=(2, 2), activation='relu', padding='same')(strided)
    strided = layers.Cropping2D(cropping=((0, 1), (0, 1)))(strided)

    both = layers.concatenate([pooled, strided], axis=3)

    pooled = layers.Conv2D(64, 3, activation='relu', padding='same')(both)
    pooled = layers.Conv2D(64, 3, activation='relu', padding='same')(pooled)
    pooled = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pooled)

    pooled = layers.Conv2D(64, 3, activation='relu', padding='same')(pooled)
    pooled = layers.Conv2D(64, 3, activation='relu', padding='same')(pooled)
    pooled = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pooled)

    strided = layers.Conv2D(64, 7, strides=(2, 2), activation='relu', padding='same')(both)
    strided = layers.Conv2D(64, 7, strides=(2, 2), activation='relu', padding='same')(strided)
    strided = layers.Cropping2D(cropping=((0, 1), (0, 1)))(strided)

    both = layers.concatenate([pooled, strided], axis=3)

    out = layers.Conv2D(128, 4, activation='relu')(both)
    out = layers.Conv2D(256, 4, activation='relu')(out)

    out = layers.Flatten()(out)

    out = layers.Dense(args.latent_size + args.latent_size)(out)

    model = keras.Model(inputs=img_input, outputs=out, name='inference_model')

    return model


class CVAE(tf.keras.Model):
    def __init__(self, args):
        super(CVAE, self).__init__()
        self.latent_size = args.latent_size
        self.inference_net = make_inference_model(args)
        self.generative_net = make_generator_model(args)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(args.batch_size, args.latent_size))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


# ****************** training and helper functions  *****************

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train(args):
    args.checkpoint.restore(args.manager.latest_checkpoint)
    if args.manager.latest_checkpoint:
        print("Restored model from {}".format(args.manager.latest_checkpoint))
    else:
        print("Initializing training from scratch.")

    epochs_so_far = args.checkpoint.step
    epochs_to_reach = epochs_so_far + args.epochs
    print("Training from epoch no. ", epochs_so_far, " to epoch no. ", epochs_to_reach)

    for epoch in tf.range(epochs_so_far, epochs_to_reach):
        start = time.time()

        for image_batch in args.train_dataset:
            compute_apply_gradients(args.cvae, image_batch, args.optimizer)
            args.cvae_loss(compute_loss(args.cvae, image_batch))

        # Produce images for the GIF as we go
        generate_and_save_images(args.cvae.generative_net,
                                 args.seed,
                                 args.nr_eval_images,
                                 epoch + 1,
                                 os.path.join(args.output_dir, "results/"))

        # Generate plots
        with args.cvae_summary_writer.as_default():
            tf.summary.scalar(name='cvae loss', data=args.cvae_loss.result(), step=epoch)
        
        args.checkpoint.step.assign_add(1)
        args.manager.save()

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        args.cvae_loss.reset_states()
