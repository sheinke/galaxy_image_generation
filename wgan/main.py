
import tensorflow as tf
import os
import argparse
import datetime
import pandas as pd
from tensorflow.keras.models import load_model

import sys
sys.path.append("..")

from utils.data_utils import DataUtilsConfig, load_data, create_animation
from WGAN_GP import *

'''
This is a basic implementation of a gradient-penalized Wasserstein GAN.
It operates on full-size images and has a default latent dimensionality of 200.
By default, we update the discriminator 5 times more often than the generator.
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # hyperparameters for architecture and training
    parser.add_argument('--epochs', type=int, help='number of epochs to train the model', default=10)
    parser.add_argument('--noise_dim', type=int, default=200, help='dimension of latent encoding vector')
    parser.add_argument('--batch_size', type=int, default=8, help='# images per batch')
    parser.add_argument('--buffer_size', type=int, default=256, help='# images to buffer')
    parser.add_argument('--nr_eval_images', type=int, default=4, help='number of evaluation images to be generated')
    parser.add_argument('--n_update_disc', type=int, default=5, help='how much more often the discriminator should be updated')
    parser.add_argument('--gp_lambda', type = float, default = 10, help = 'lambda of gradient penalty')
    parser.add_argument('--img_size', type=int, default=1000,
                        help='Image size. Changing this parameter will require changes in the network architectures (see WGAN_GP.py')
    # input- / output-directories
    parser.add_argument('--data_root', type=str, default='/cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/', help='path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./')
    args = parser.parse_args()

    config = DataUtilsConfig()
    config.data_root = args.data_root
    config.batch_size = args.batch_size
    config.buffer_size = args.buffer_size
    config.img_size = args.img_size
    config.score_threshold = 1.0
    config.score_for_actual_labeled = 1.1  # make sure to include labeled subset

    # initialize output directories:
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

    # build dataset
    args.train_dataset = load_data(config)
    # don't need scores
    args.train_dataset = args.train_dataset.map(lambda img, score: img)

    # build generator
    args.generator = build_generator(args)

    # build discriminator
    args.discriminator = build_discriminator(args)

    # optimizers 
    args.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    args.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

    # define checkpoint
    args.checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                          generator_optimizer=args.generator_optimizer,
                                          discriminator_optimizer=args.discriminator_optimizer,
                                          generator=args.generator,
                                          discriminator=args.discriminator)
    args.manager = tf.train.CheckpointManager(args.checkpoint, os.path.join(args.output_dir, "checkpoints"), max_to_keep=3)

    # we will reuse this seed over time to visualize progress in the animated GIF
    args.seed = tf.random.normal([args.nr_eval_images, args.noise_dim])

    # set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.generator_log_dir = os.path.join("logs/", current_time, 'generator/')
    args.discriminator_log_dir = os.path.join("logs/", current_time, 'discriminator/')
    args.adv_loss_log_dir = os.path.join("logs/", current_time, 'adv_loss/')
    args.generator_summary_writer = tf.summary.create_file_writer(args.generator_log_dir)
    args.discriminator_summary_writer = tf.summary.create_file_writer(args.discriminator_log_dir)
    args.adv_loss_summary_writer = tf.summary.create_file_writer(args.adv_loss_log_dir)

    # initialize Metrics
    args.adv_loss = tf.keras.metrics.Mean(name='Adversarial_Loss')
    args.gen_loss = tf.keras.metrics.Mean(name='Generator_Loss')
    args.disc_loss = tf.keras.metrics.Mean(name='Discriminator_Loss')

    # train GAN
    train(args)
    # save GIF 
    create_animation(input_dir=os.path.join(args.output_dir, "results/"),
                 anim_out_file=os.path.join(args.output_dir, "results/animation.gif"))
    # save models as H5
    args.discriminator.save(os.path.join(args.output_dir, "checkpoints", 'wgan_trained_discriminator.h5'))
    args.generator.save(os.path.join(args.output_dir, "checkpoints", 'wgan_trained_generator.h5'))
