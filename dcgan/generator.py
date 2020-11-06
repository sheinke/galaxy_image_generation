'''
Generator model used in the DCGAN.
'''
import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model(img_width, img_height, noise_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(125 * 125 * 2, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((125, 125, 2)))
    # Note: None is the batch size
    assert model.output_shape == (None, 125, 125, 2)

    model.add(layers.Conv2DTranspose(
        4, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 125, 125, 4)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        8, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 125, 125, 8)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        16, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 125, 125, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 250, 250, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 500, 500, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, img_width, img_height, 1)

    return model

