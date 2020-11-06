
import argparse
import os

import tensorflow as tf 


from WGAN_GP import build_discriminator, build_generator

'''
This script allows to convert Tensorflow CheckpointManager's checkpoints to H5 checkpoints.
Given a directory with valid Tensorflow checkpoints, it will try to restore said checkpoints
and save both models (i.e. the discriminator and the generator) as H5 files inside the
checkpoint directory.
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # hyperparameters for architecture and training
    parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint', default='./checkpoints/')
    parser.add_argument('--noise_dim', type=int, default=200, help='dimension of latent encoding vector')
    parser.add_argument('--img_size', type=int, default=1000, help='image size')
    args = parser.parse_args()

    # build discriminator
    args.discriminator = build_discriminator(args)
    # build generator
    args.generator = build_generator(args)

    # optimizers 
    args.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    args.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

    # define checkpoint
    args.checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                          generator_optimizer=args.generator_optimizer,
                                          discriminator_optimizer=args.discriminator_optimizer,
                                          generator=args.generator,
                                          discriminator=args.discriminator)
    args.manager = tf.train.CheckpointManager(args.checkpoint, args.checkpoint_path, max_to_keep=3)

    args.checkpoint.restore(args.manager.latest_checkpoint)
    if args.manager.latest_checkpoint:
        print("Restored model from {}".format(args.manager.latest_checkpoint))
    else:
        print("ERROR: Could not find any stored checkpoints. Exiting...")
        quit(-1)

    # save as H5
    args.discriminator.save(os.path.join(args.checkpoint_path, 'wgan_trained_discriminator.h5'))
    args.generator.save(os.path.join(args.checkpoint_path, 'wgan_trained_generator.h5'))
