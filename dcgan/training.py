'''
Training script:

    1. GAN training
    2. Mapper model training
    
'''

import tensorflow as tf
import time
import sys
sys.path.append("..")
from utils.data_utils import generate_and_save_images

@tf.function
def train_step(images, labels, batch_size, noise_dim, generator, discriminator, generator_optimizer,
               discriminator_optimizer, gen_loss_metric, disc_loss_metric):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, labels)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))

        gen_loss_metric(gen_loss)
        disc_loss_metric(disc_loss)

def train(train_dataset, epochs, batch_size, noise_dim, generator, discriminator, generator_optimizer,
          discriminator_optimizer,
          gen_loss, disc_loss, manager, checkpoint, generator_summary_writer, discriminator_summary_writer,
          seed, result_path, nr_eval_images):
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored model from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    epochs_so_far = checkpoint.step
    epochs = epochs + epochs_so_far

    for epoch in tf.range(epochs_so_far, epochs):
        start = time.time()

        for image_batch, labels in train_dataset:
            train_step(image_batch,
                       labels,
                       batch_size,
                       noise_dim,
                       generator,
                       discriminator,
                       generator_optimizer,
                       discriminator_optimizer,
                       gen_loss,
                       disc_loss)

        # Produce images for the GIF as we go
        generate_and_save_images(generator, seed, nr_eval_images, epoch + 1, result_path)

        # Generate plots
        with generator_summary_writer.as_default():
            tf.summary.scalar(name='generator loss', data=gen_loss.result(), step=epoch)
        with discriminator_summary_writer.as_default():
            tf.summary.scalar(name='discriminator loss', data=disc_loss.result(), step=epoch)

        # Save the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        #   manager.save

        checkpoint.step.assign_add(1)
        manager.save()


        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        disc_loss.reset_states()
        gen_loss.reset_states()

    # Generate after the final epoch
    generate_and_save_images(generator, seed, nr_eval_images, epochs, result_path)


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output, real_labels):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def mapper_train_step(images, labels, batch_size, mapper, mapper_optimizer, discriminator, map_train_loss_metric):
    
    real_output = discriminator(images, training=False)
    
    with tf.GradientTape() as map_tape:

        # learn a function that maps the output of the discriminator to the real range of scores
        predicted_score_real = mapper(real_output, training=True)
        map_loss = mapper_loss(predicted_score_real, labels)

        # compute gradients of mapper
        gradients_of_mapper = map_tape.gradient(
            map_loss, mapper.trainable_variables)

        # apply gradients of mapper 
        mapper_optimizer.apply_gradients(
            zip(gradients_of_mapper, mapper.trainable_variables))

        map_train_loss_metric(map_loss)



def train_mapper(dataset, epochs, batch_size, mapper, mapper_optimizer, map_train_loss_metric, map_val_loss_metric,
                 mapper_train_summary_writer, mapper_val_summary_writer, discriminator, manager, checkpoint):

    # 9600 scored images in total 
    # after applying the threshold 230 batches

    size = 230

    train_size = int(0.8 * size)
    val_size = int(0.2 * size)
    
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored model from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    epochs_so_far = checkpoint.step
    epochs = epochs + epochs_so_far

    for epoch in tf.range(epochs_so_far, epochs):
        start = time.time()

        # training
        for image_batch, labels in train_dataset:
            mapper_train_step(image_batch,
                              labels,
                              batch_size,
                              mapper,
                              mapper_optimizer,
                              discriminator,
                              map_train_loss_metric)

        # Generate plots
        with mapper_train_summary_writer.as_default():
            tf.summary.scalar(name='mapper train loss', data=map_train_loss_metric.result(), step=epoch)

        map_train_loss_metric.reset_states()

        # validation
        if epoch % 2 == 0:

            print(f'validation step')
            for image_batch, labels in test_dataset:
                real_output = discriminator(image_batch, training=False)
                predicted_score_real = mapper(real_output, training=False)
                map_loss = mapper_loss(predicted_score_real, labels)
                map_val_loss_metric(map_loss)

            # Generate plots
            with mapper_val_summary_writer.as_default():
                tf.summary.scalar(name='mapper val loss', data=map_val_loss_metric.result(), step=epoch)

        map_val_loss_metric.reset_states()

        checkpoint.step.assign_add(1)
        manager.save()

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

def mapper_loss(predicted_scores, real_labels):
    
    return tf.keras.losses.MSE(real_labels, predicted_scores)
