
import argparse
import os
import numpy as np 
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

'''
This script allows to generate pictures, using a specified generator model.
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_images', type=int, help='number of images to generate', default=100)
    parser.add_argument('--batch_size', type=int, default=8, help='# images per batch')
    parser.add_argument('--generator_path', type=str, default="./checkpoints/wgan_trained_generator.h5", help='path to trained generator model')
    parser.add_argument('--noise_dim', type=int, default=200, help='dimension of latent encoding vector. Has to match the input format of the specified generator model')
    parser.add_argument('--output_dir', type=str, default='./results/full_images/')    
    args = parser.parse_args()

    # initialize output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load model 
    assert args.generator_path.endswith('h5'), "Not a valid H5 file: {}".format(args.generator_path)
    model = load_model(args.generator_path, compile=False)

    # in order to avoid OOM exceptions, we iterate over smaller groups of pictures
    num_full_batches = args.n_images // args.batch_size
    group_sizes = [args.batch_size] * num_full_batches

    num_remaining_imgs = args.n_images % args.batch_size
    group_sizes += [num_remaining_imgs]

    img_counter = 0
    for group_idx in range(len(group_sizes)):
        # generate seed for the latent space
        seed = tf.random.normal([group_sizes[group_idx], args.noise_dim])
        # generate pictures
        generated_images = model(seed, training=False)
        # save pictures to disk
        for i in range(generated_images.shape[0]):
            img = generated_images[i, :, :, 0]
            img = np.clip(img, 0.0, 1.0)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(os.path.join(args.output_dir, str(img_counter) + ".png"), optimize=False, compress_level=0)
            img_counter += 1
