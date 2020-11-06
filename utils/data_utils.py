"""
Data utils.

Example dataset loading with default values for augmentation:
from data_utils import load_data, DataUtilsConfig
DataConfig = DataUtilsConfig()
DataConfig.data_root = path_to_cil_dataset
ds = load_data(DataConfig)

Augmentation settings can be changed by modifying the DataUtilsConfig object.
Please see  DataUtilsConfig class definition.
"""

import imageio
import glob
import pandas as pd
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import argparse
import random
import scipy.ndimage as ndimage
#from sky_ds import sky_ds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
AUTOTUNE = tf.data.experimental.AUTOTUNE


# ****************** output image visualisation, postprocessing  *****************

def visualize(image):
    """
    visualises the image
    :param image: [0,1] float image of size [img_height, img_width]
    """

    print("image shape" + str(image.shape))
    pil_img = Image.fromarray((255 * image).astype(np.uint8))
    pil_img.show()



def create_animation(input_dir, anim_out_file):
    """
    Creates an animated GIF from all png files called "image*" that are
    located in input_dir. Uses alphanumerical ordering of input files.
    :param input_dir: Input directory that contains several png imgages.
    :param anim_out_file: Output GIF file (should have correct file ending).
    """
    with imageio.get_writer(anim_out_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(input_dir, 'image*.png'))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)


def generate_and_save_images(model, seed, n_images_to_generate, epoch, output_dir):
    """
    Generates and saves a set of images, using the given generative model.
    This function is mainly meant to be used for visualizing progress with a GIF.
    The dimensionality of seed has to match the model input as well as the number of
    images to generate (i.e. n_images_to_generate). Images are named after the epoch
    value and saved into output_dir.
    :param model: The model that is used to create the images.
    :param seed: The input seed (should be fixed across epochs for usable results).
    :param n_images_to_generate: You guessed it.
    :param epoch: The epoch number. Images are named using this value.
    :param output_dir: Images will be saved under this directory.
    """

    # notice `training` is set to False, thus all layers run in inference mode
    pictures = model(seed, training=False)

    fig = plt.figure(figsize=(8, 8))

    grid_length = int(math.ceil(np.sqrt(n_images_to_generate)))

    for i in range(pictures.shape[0]):
        plt.subplot(grid_length, grid_length, i + 1)
        plt.imshow(pictures[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(output_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close(fig)


def crop_stylegan(save_path, input, k=0, size_in=1024, size_out=1000):
    """
    Useful for postprocessing of StyleGAN output.
    Crops an 8x8 image into 64 smaller ones.
    :param save_path: path where images should be stored
    :param input: input image path
    :param k: index of image, save paths will add indices from k to k+63
    :param height: image height
    :param width: image width
    """
    im = Image.open(input)
    im_name = input.split("/")[-1].split(".")[0]
    imgwidth, imgheight = im.size
    diff = (size_in - size_out) / 2
    for i in range(0, imgheight, size_in):
        for j in range(0, imgwidth, size_in):
            box = (j + diff, i + diff, j + size_in - diff, i + size_in - diff)
            a = im.crop(box)
            a.save(os.path.join(save_path, "post_{}_{}.png".format(im_name, k)))
            k += 1

def stylegan_postprocess(eval_img_path, save_path):
    """
    Takes generated StyleGAN images and splits them into 1000x1000 parts.
    :param eval_img_path: path to generated images
    :param save_path: path for post-processed images
    """
    # take images with numbers 10-29
    image_paths = glob.glob(eval_img_path + "/i[0-2][0-9].png")
    os.mkdir(save_path)
    print(image_paths)
    for path in image_paths:
        crop_stylegan(save_path, path)


# ****************** image loading, augmentation  *****************

def rotate(image):
    """
    :param image: image to be rotated
    :return: list of images containing the input image rotated by 0, 90, 180 and 270 degrees.
    """
    rot0 = image
    rot1 = tf.image.rot90(rot0)
    rot2 = tf.image.rot90(rot1)
    rot3 = tf.image.rot90(rot2)
    return [rot0, rot1, rot2, rot3]


def rotate_image_scipy(image, angle):
    """
    :param image: image to be rotated
    :param angle: rotation angle
    :return: rotated angle
    """
    image = ndimage.rotate(image, angle, reshape=False)
    return image


def random_rotate_image(image):
    """
    :param image: image to be rotated
    :return: image rotated by a random angle
    """
    angle = random.randint(0, 360)
    [rot_image, ] = tf.py_function(rotate_image_scipy, [image, angle], [tf.float32])
    rot_image.set_shape(image.shape)
    return rot_image


def mirror(image):
    """
    :param image: image to be flipped
    :return: y axis flipped image
    """
    return tf.image.flip_left_right(image)


def img_subtract_mean(image):
    """
    :param image: image
    :return: image with subtracted mean
    """
    return tf.maximum(tf.zeros_like(image), image - tf.math.reduce_mean(image, keepdims=True))


def img_binarise(image):
    """
    :param image: image
    :return: image with 0 and 1 values only
    """
    img_zero_avg = img_subtract_mean(image)
    return tf.where(img_zero_avg >= 1.1 / 255, tf.ones_like(image), tf.zeros_like(image))


def random_crop(image, img_size: int, pad_size: int):
    """
    Produces a random crop of the input image.
    :param image: Quadratic input image.
    :param img_size: Size of the input image. This is also the size of the output image.
    :param pad_size: Number of padding pixels.
    """
    padded_size = img_size + pad_size
    image = tf.image.resize_with_crop_or_pad(image, padded_size, padded_size)
    image = tf.image.random_crop(image, size=[img_size, img_size, 1])

    return image


def load_img(file_path, img_size: int, resize_method='bilinear'):
    """
    Decodes an image file and resizes it to given image size.
    Pixel values are scaled to the [0; 1] interval.
    :param file_path: The input file path.
    :param img_size: Desired size of the quadratic output image.
    :param resize_method: How to resize images. Options are 'bilinear' and 'padded'.
    """
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=1)
    # scale image to [0; 1]
    img = tf.image.convert_image_dtype(img, tf.float32)

    # resize the image to the desired size.
    if resize_method == 'bilinear':
        return tf.image.resize(img, [img_size, img_size])
    elif resize_method == 'padded':
        return tf.image.resize_with_crop_or_pad(img, img_size, img_size)
    else:
        print("Critical: unknown resize method in data_utils.load_img")
        quit(-1)


def augment_img(img, score, img_size=1000, pad_size=100, do_random_crop=True, nr_shifts=2):
    """
    Augments the given sample. Image dimensions are kept.
    :param img: Input image.
    :param score: Score value for given input image.
    :param img_size: Size of the input image
    :param pad_size: Amount of padding.
    :param do_random_crop: True if random crop augmentation should be done
    :param nr_shifts: number of cropped and shifted images
    """

    imgs_flip = [img, mirror(img)]

    imgs_rot = []
    for img in imgs_flip:
        imgs_rot += rotate(img)

    result = imgs_rot

    if do_random_crop:
        result = []
        for img in imgs_rot:
            rand_crops = [random_crop(img, img_size, pad_size) for _ in range(nr_shifts)]
            result += rand_crops

    return tf.data.Dataset.from_tensor_slices(result).map(lambda m_img: (m_img, score))

def get_img(path,nr_crops=2000):
    #read imge
    img = tf.io.read_file(path)
    

    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=1)
    
    #img = (tf.image.convert_image_dtype(img, tf.float32)/256)
    img = (tf.image.convert_image_dtype(img, tf.float32))

    crops = [(tf.image.random_crop(img, size=[1024, 1024, 1])) for _ in range(nr_crops)]

    

    return tf.data.Dataset.from_tensor_slices(crops).map(lambda m_img: (m_img, -1))

def sky_ds(picture_path = f'./out/'):

    ds = tf.data.Dataset.list_files(f"{picture_path}*.png")

    #show one of the pictures used
    for batch in ds:
        print(batch)
        break

    #get images
    ds = ds.flat_map(get_img)

    return ds

# ****************** dataset loading interface, usage example  *****************

class DataUtilsConfig:
    """
    Configuration class passed to load_data()
    """
    data_root = None  # directory where scored, labeled, scored.csv, labeled.csv files exist
    batch_size = 8
    buffer_size = 128

    score_threshold = 1.25  # take images with score greater or equal that this
    load_only_scored = False  # use only scored images
    score_for_actual_labeled = 1.25  # score value for labeled actual galaxies

    do_random_crop = True  # shifts and crop
    nr_shifts = 2  # nr of shifted images per each image
    img_size = 1000  # image size: images are resized to this size after decoding
    resize_method = 'bilinear'  # options: {'bilinear', 'padded'}
    pad_size = 100  # amount of padding

    img_subtract_mean = False  # make images to have 0 average
    img_binarise = False  # use images with 0 or 1 values only
    img_random_rotate = False  # randomly rotate images

    sky_ds = False # use crawled sky image dataset


def load_data(config):
    """
    :param config: DataUtilsConfig object
    :return: tf.data.Dataset with augmented images and scores

    usage with default values for augmentation:

    from data_utils import load_data, DataUtilsConfig
    DataConfig = DataUtilsConfig()
    DataConfig.data_root = path_to_cil_dataset
    ds = load_data(DataConfig)

    """

    if config.sky_ds:
        ds = sky_ds(config.data_root)
    else:
        print("Loading data from: {}".format(config.data_root))
        labeled_file_path = os.path.join(config.data_root, "labeled.csv")
        labeled_input_dir = os.path.join(config.data_root, "labeled/")
        scored_file_path = os.path.join(config.data_root, "scored.csv")
        scored_input_dir = os.path.join(config.data_root, "scored/")

        # get labels and labels filepaths
        labels = pd.read_csv(labeled_file_path)
        labels_filepaths = [labeled_input_dir + str(id_) + '.png' for id_ in labels['Id'].to_numpy()]

        # get scores and scores filepaths
        scores = pd.read_csv(scored_file_path)
        scores_filepaths = [scored_input_dir + str(id_) + '.png' for id_ in scores['Id'].to_numpy()]

        if config.load_only_scored:
            ds_name = tf.data.Dataset.from_tensor_slices(np.array(scores_filepaths))
            scores = scores['Actual'].to_numpy()
            ds_scores = tf.data.Dataset.from_tensor_slices(scores)

        else:
            filepaths = labels_filepaths + scores_filepaths
            # create label and scored filename dataset
            ds_name = tf.data.Dataset.from_tensor_slices(np.array(filepaths))
            # convert labels to scores
            labels = labels['Actual'].to_numpy() * config.score_for_actual_labeled
            scores = scores['Actual'].to_numpy()
            ds_scores = tf.data.Dataset.from_tensor_slices(np.concatenate([labels, scores]))

        # join filepaths and scores together
        ds = tf.data.Dataset.zip((ds_name, ds_scores))

        # filter out based on score threshold
        ds = ds.filter(lambda name, score: score >= config.score_threshold)

        # load images
        ds = ds.map(lambda name, score: (load_img(name, config.img_size, config.resize_method), score),
                    num_parallel_calls=AUTOTUNE)

        #  augment images
        ds = ds.flat_map(lambda img, score: augment_img(img=img,
                                                        score=score,
                                                        img_size=config.img_size,
                                                        pad_size=config.pad_size,
                                                        do_random_crop=config.do_random_crop,
                                                        nr_shifts=config.nr_shifts))

    if config.img_random_rotate:
        ds = ds.map(lambda image, score: (random_rotate_image(image), score))

    # subtract image mean
    if config.img_subtract_mean:
        ds = ds.map(lambda image, score: (img_subtract_mean(image), score))

    # binarise images
    elif config.img_binarise:  # else-if so we do not execute img_subtract_mean twice
        ds = ds.map(lambda image, score: (img_binarise(image), score))

    # shuffle and batch
    ds = ds.shuffle(config.buffer_size).batch(config.batch_size).prefetch(buffer_size=AUTOTUNE)

    print("Dataset generated.")

    return ds


def load_query_data(config):
    """
    Loads query data from the query directory.
    Note that this function only works for the designated query directory of the provided dataset;
    if you want to load an arbitrary list of png files as a dataset, please see the judge script.
    :param config: Configuration object, has to hold valid data_root, img_size, resize_method
    """

    # get a list of all png file names inside the query directory
    query_dir = join(config.data_root, "query/")
    query_pics_files = [f for f in listdir(query_dir) if isfile(join(query_dir, f)) and f.endswith("png")]
    # cut off the ".png", take the remaining file names as IDs
    query_pics_ids = [int(f[:-4]) for f in query_pics_files]
    # fully extend file names
    query_pics_files = [join(query_dir, f) for f in query_pics_files]

    ds_name = tf.data.Dataset.from_tensor_slices(np.array(query_pics_files))
    ds_ids = tf.data.Dataset.from_tensor_slices(np.array(query_pics_ids))

    ds_picture = ds_name.map(lambda x: load_img(x, config.img_size, config.resize_method), num_parallel_calls=AUTOTUNE)
    ds = tf.data.Dataset.zip((ds_picture, ds_ids))

    return ds.batch(config.batch_size).prefetch(buffer_size=AUTOTUNE)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str,
                        default='/cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/',
                        help='path to the dataset')
    opt = parser.parse_args()
    DataConfig = DataUtilsConfig()
    DataConfig.data_root = opt.data_root
    DataConfig.batch_size = 4
    DataConfig.img_binarise = 1
    dataset = load_data(DataConfig)

    for img, label in dataset:
        print(img.numpy().shape)
        visualize(img.numpy()[0, :, :, 0])
        break
