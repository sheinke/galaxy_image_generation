"""
The code is based on the following implementation: https://github.com/WangZesen/WGAN-GP-Tensorflow-v2
"""
import time
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
sys.path.append("..")

from utils.data_utils import generate_and_save_images


def build_generator(args):
    latent_input = keras.Input(shape=(args.noise_dim,))

    main = layers.Dense(8 * 8 * 128, activation='relu', use_bias=False)(latent_input)
    main = layers.Reshape((8, 8, 128))(main)
    
    strided = layers.Conv2DTranspose(128, 5, strides=(2, 2), activation='relu', padding='same')(main)
    strided = layers.Conv2DTranspose(64, 5, strides=(2, 2), activation='relu', padding='same')(strided)
    strided = layers.Conv2DTranspose(64, 5, strides=(2, 2), activation='relu', padding='same')(strided)
    strided = layers.Conv2DTranspose(32, 7, strides=(2, 2), activation='relu', padding='same')(strided)
    strided = layers.Cropping2D(cropping=((1, 2), (1, 2)))(strided)  # 125 x 125
    strided = layers.Conv2DTranspose(16, 7, strides=(2, 2), activation='relu', padding='same')(strided)
    strided = layers.Conv2DTranspose(8, 7, strides=(2, 2), activation='relu', padding='same')(strided)

    out = layers.Conv2DTranspose(1, 7, strides=(2, 2), activation='sigmoid', padding='same')(strided)

    model = keras.Model(inputs=latent_input, outputs=out, name='generator_model')
    assert model.output_shape == (None, args.img_size, args.img_size, 1)

    return model


def build_discriminator(args):
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

    pooled = layers.Conv2D(128, 3, activation='relu', padding='same')(pooled)
    pooled = layers.Conv2D(128, 3, activation='relu', padding='same')(pooled)
    pooled = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pooled)

    strided = layers.Conv2D(128, 7, strides=(2, 2), activation='relu', padding='same')(both)
    strided = layers.Conv2D(128, 7, strides=(2, 2), activation='relu', padding='same')(strided)
    strided = layers.Cropping2D(cropping=((0, 1), (0, 1)))(strided)

    both = layers.concatenate([pooled, strided], axis=3)

    out = layers.Flatten()(both)

    out = layers.Dense(1)(out)

    model = keras.Model(inputs=img_input, outputs=out, name='inference_model')

    return model


@tf.function
def train_step_gen(batch_size, noise_dim, generator, discriminator, gen_opt, gen_loss):
    with tf.GradientTape() as tape:
        z = tf.random.uniform([batch_size, noise_dim], -1.0, 1.0)
        fake_sample = generator(z)
        fake_score = discriminator(fake_sample)
        loss = - tf.reduce_mean(fake_score)
    gradients = tape.gradient(loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients, generator.trainable_variables))
    gen_loss(loss)


@tf.function
def train_step_dis(real_sample, noise_dim, gen, dis, disc_opt, gp_lambda, disc_loss, adv_loss):
    batch_size = real_sample.get_shape().as_list()[0]
    with tf.GradientTape() as tape:
        z = tf.random.uniform([batch_size, noise_dim], -1.0, 1.0)
        fake_sample = gen(z)
        real_score = dis(real_sample)
        fake_score = dis(fake_sample)

        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        inter_sample = fake_sample * alpha + real_sample * (1 - alpha)
        with tf.GradientTape() as tape_gp:
            tape_gp.watch(inter_sample)
            inter_score = dis(inter_sample)
        gp_gradients = tape_gp.gradient(inter_score, inter_sample)
        gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis = [1, 2, 3]))
        gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)
        
        loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + gp * gp_lambda
        
    gradients = tape.gradient(loss, dis.trainable_variables)
    disc_opt.apply_gradients(zip(gradients, dis.trainable_variables))

    disc_loss(loss)
    adv_loss(loss - gp * gp_lambda)


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

        counter = 0
        for image_batch in args.train_dataset:
            counter += 1
            if counter % args.n_update_disc != 0:
                train_step_dis(image_batch,
                               args.noise_dim,
                               args.generator,
                               args.discriminator,
                               args.discriminator_optimizer,
                               args.gp_lambda,
                               args.disc_loss,
                               args.adv_loss)
            else:
                train_step_gen(args.batch_size,
                               args.noise_dim,
                               args.generator,
                               args.discriminator,
                               args.generator_optimizer,
                               args.gen_loss)
                
        # Produce images for the GIF as we go
        generate_and_save_images(args.generator,
                                 args.seed,
                                 args.nr_eval_images,
                                 epoch + 1,
                                 os.path.join(args.output_dir, "results/"))

        # Generate plots
        with args.generator_summary_writer.as_default():
            tf.summary.scalar(name='generator loss', data=args.gen_loss.result(), step=epoch)
        with args.discriminator_summary_writer.as_default():
            tf.summary.scalar(name='discriminator loss', data=args.disc_loss.result(), step=epoch)
        with args.adv_loss_summary_writer.as_default():
            tf.summary.scalar(name='adversarial loss', data=args.adv_loss.result(), step=epoch)

        args.checkpoint.step.assign_add(1)
        args.manager.save()

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        args.disc_loss.reset_states()
        args.adv_loss.reset_states()
        args.gen_loss.reset_states()
