
import tensorflow as tf
import os
import argparse
import datetime
import pandas as pd

import sys
sys.path.append("..")

from utils.data_utils import DataUtilsConfig, load_data, create_animation
from CVAE import *


'''
This is a simple implementation of a Convolutional Variational Autoencoder CVAE,
adapted from www.tensorflow.org/tutorials/generative/cvae.
It uses 1000x1000 images and a 200-dimensional latent space.
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # hyperparameters for architecture and training
    parser.add_argument('--epochs', type=int, help='number of epochs to train the model', default=50)
    parser.add_argument('--latent_size', type=int, default=200, help='dimension of latent encoding vector')
    parser.add_argument('--batch_size', type=int, default=8, help='# images per batch')
    parser.add_argument('--buffer_size', type=int, default=128, help='# images to buffer')
    parser.add_argument('--nr_eval_images', type=int, default=4, help='number of evaluation images to be generated')
    parser.add_argument('--img_size', type=int, default=1000, help='image size')
    # input- / output-directories
    parser.add_argument('--data_root', type=str, default='/cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/', help='path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./')
    args = parser.parse_args()

    config = DataUtilsConfig()
    config.data_root = args.data_root
    config.batch_size = args.batch_size
    config.buffer_size = args.buffer_size
    config.img_size = args.img_size
    config.score_threshold = 0.3

    # initialize output directories:
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

    # build dataset
    args.train_dataset = load_data(config)
    # don't need scores
    args.train_dataset = args.train_dataset.map(lambda img, score: img)
    # build network
    args.cvae = CVAE(args)    
    # optimizer
    args.optimizer = tf.keras.optimizers.Adam(1e-4)
    # define checkpoint
    args.checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                          cvae_optimizer=args.optimizer,
                                          cvae=args.cvae)
    args.manager = tf.train.CheckpointManager(args.checkpoint,
                                              os.path.join(args.output_dir, "checkpoints"),
                                              max_to_keep=30)
    # we will reuse this seed over time to visualize progress in the animated GIF
    args.seed = tf.random.normal([args.nr_eval_images, args.latent_size])
    # set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.cvae_log_dir = os.path.join(args.output_dir, "logs", current_time, 'cvae/')
    args.cvae_summary_writer = tf.summary.create_file_writer(args.cvae_log_dir)
    # initialize Metrics
    args.cvae_loss = tf.keras.metrics.Mean(name='CVAE_Loss')
    # train GAN
    train(args)  
    # save GIF 
    create_animation(input_dir=os.path.join(args.output_dir, "results/"),
                     anim_out_file=os.path.join(args.output_dir, "results/animation.gif"))
