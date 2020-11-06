'''
Main function used to orchestrate the entire training of a DCGAN
'''

# temporary solution to disable png warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import os
import argparse
import datetime

import sys
sys.path.append("..")

from utils.data_utils import DataUtilsConfig, load_data, create_animation
from generator import make_generator_model
from discriminator import make_discriminator_model
from training import train

if __name__ == "__main__":

      parser = argparse.ArgumentParser(description='')
      
      # hyperparameters for architecture and training
      parser.add_argument('--epochs', type=int, help='number of epochs to train the model', default=100)
      parser.add_argument('--batch_size', type=int, default=16, help='# images per batch')
      parser.add_argument('--buffer_size', type=int, default=128, help='# images to buffer')
      parser.add_argument('--nr_eval_images', type=int, default=4, help='number of evaluation images to be generated')
      parser.add_argument('--img_size', type=int, default=1000, help='image size')
      # input- / output-directories
      parser.add_argument('--data_root', type=str, default='/cluster/scratch/pblatter/cil/cosmology_aux_data_170429/cosmology_aux_data_170429/', help='path to the dataset')
      parser.add_argument('--output_dir', type=str, default='./')
      parser.add_argument('--noise_dim', type=int, default=100, help='dimension of noise input vector')
      parser.add_argument('--img_width', type=int, default=1000, help='image width')
      parser.add_argument('--img_height', type=int, default=1000, help='image height')

      
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
      train_dataset = load_data(config)

      # build generator
      generator = make_generator_model(args.img_width, args.img_height, args.noise_dim)

      # build discriminator
      discriminator = make_discriminator_model(args.img_width, args.img_height)

      # optimizers
      generator_optimizer = tf.keras.optimizers.Adam(1e-4)
      #discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
      discriminator_optimizer = tf.keras.optimizers.SGD(1e-4)
      mapper_optimizer = tf.keras.optimizers.Adam(1e-4)

      # define checkpoint
      checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                          generator_optimizer=generator_optimizer,
                                          discriminator_optimizer=discriminator_optimizer,
                                          generator=generator,
                                          discriminator=discriminator)

      # checkpoint manager
      manager = tf.train.CheckpointManager(checkpoint,
                                           os.path.join(args.output_dir, "checkpoints"),
                                           max_to_keep=3)

      # We will reuse this seed overtime (so it's easier)
      # to visualize progress in the animated GIF)
      seed = tf.random.normal([args.nr_eval_images, args.noise_dim])

      # build results directory if non-existent yet
      result_path = os.path.join(args.output_dir, "results/")

      # set up summary writers
      current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      generator_log_dir = os.path.join(args.output_dir, "logs") + '/generator'
      discriminator_log_dir = os.path.join(args.output_dir, "logs") + '/discriminator'
      generator_summary_writer = tf.summary.create_file_writer(generator_log_dir)
      discriminator_summary_writer = tf.summary.create_file_writer(discriminator_log_dir)

      # Initialize Metrics
      gen_loss = tf.keras.metrics.Mean(name='Generator_Loss')
      disc_loss = tf.keras.metrics.Mean(name='Discriminator_Loss')
      
      # train GAN
      train(train_dataset, args.epochs, args.batch_size, args.noise_dim, generator, discriminator,
            generator_optimizer, discriminator_optimizer, gen_loss, disc_loss,
            manager, checkpoint, generator_summary_writer, discriminator_summary_writer,
            seed, result_path, args.nr_eval_images)

      # save GIF
      # save GIF 
      create_animation(input_dir=os.path.join(args.output_dir, "results/"),
                     anim_out_file=os.path.join(args.output_dir, "results/animation.gif"))