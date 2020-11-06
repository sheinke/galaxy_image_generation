
import os
import argparse
import random
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

import sys
sys.path.append("..")

from utils.data_utils import DataUtilsConfig, load_img, augment_img, load_query_data

'''
This script builds a simple CNN and trains it on fullsize images to accurately predict score
values. This script uses separate functions for data loading, since it slightly rebalances
the score distribution and needs validation data.
'''


def load_score_prediction_data(config, val_split=0.05):
    """
    Loads a dataset that can be used for score prediction. This function is highly similar to
    data_utils.load_data, but it performs val-splitting. Since we do not always need that for
    our generative models, we decided to factor this functionality out into this function.
    :param config: DataUtilsConfig object
    :return: tf.data.Dataset with augmented images and scores
    """
    print("Loading score prediction data from: {}".format(config.data_root))
    scored_file_path = os.path.join(config.data_root, "scored.csv")
    scored_input_dir = os.path.join(config.data_root, "scored/")
    # get scores and scores filepaths
    scores = pd.read_csv(scored_file_path)
    # split into train and val set
    num_val_samples = int(len(scores) * val_split)
    scores_val = scores.iloc[0: num_val_samples].copy()
    scores_train = scores.iloc[num_val_samples: ].copy()

    mean, stdev = np.mean(scores_train['Actual']), np.std(scores_train['Actual'])
    scores_train['Actual'] = scores_train['Actual'].map(lambda x: (x - mean) / stdev)
    scores_val['Actual'] = scores_val['Actual'].map(lambda x: (x - mean) / stdev)

    # get filepaths and score values as lists:
    scores_filepaths_train = [scored_input_dir + str(id_) + '.png' for id_ in scores_train['Id'].to_numpy()]
    scores_filepaths_val = [scored_input_dir + str(id_) + '.png' for id_ in scores_val['Id'].to_numpy()]

    # convert to TF datasets and join together
    ds_name_train = tf.data.Dataset.from_tensor_slices(np.array(scores_filepaths_train))
    ds_scores_train = tf.data.Dataset.from_tensor_slices(scores_train['Actual'].to_numpy())
    
    ds_train = tf.data.Dataset.zip((ds_name_train, ds_scores_train))

    ds_name_val = tf.data.Dataset.from_tensor_slices(np.array(scores_filepaths_val))
    ds_scores_val = tf.data.Dataset.from_tensor_slices(scores_val['Actual'].to_numpy())
    ds_val = tf.data.Dataset.zip((ds_name_val, ds_scores_val))

    # filter out based on score threshold
    ds_train = ds_train.filter(lambda name, score: score >= config.score_threshold)
    ds_val = ds_val.filter(lambda name, score: score >= config.score_threshold)

    # load images
    ds_train = ds_train.map(lambda name, score: (load_img(name, config.img_size, config.resize_method), score),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(lambda name, score: (load_img(name, config.img_size, config.resize_method), score),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #  augment train set
    ds_train = ds_train.flat_map(lambda img, score: augment_img(img=img,
                                                                score=score,
                                                                img_size=config.img_size,
                                                                pad_size=config.pad_size,
                                                                do_random_crop=config.do_random_crop,
                                                                nr_shifts=config.nr_shifts))

    # shuffle and batch
    ds_train = ds_train.shuffle(config.buffer_size).batch(config.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.shuffle(config.buffer_size).batch(config.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print("Score prediction dataset generated.")

    return ds_train, ds_val, mean, stdev


def make_query_prediction(predictor, query) -> pd.DataFrame:
    """
    Feeds the provided query dataset through the predictor and converts all ID-score pairs
    into a Pandas dataframe.
    :param predictor: The model that is used to predict scores. Input dimensions have to match.
    :param query: Query dataset. This should be the result of a call to data_utils.load_query_data
    """
    results = []  # list of dictionaries
    for image_batch, ids_batch in query:
        prediction = predictor(image_batch, training=False)
        for i in range(prediction.shape[0]):  # assume channels-last format
            results.append({"Id": ids_batch.numpy()[i], "Predicted": prediction.numpy()[i, 0]})          

    df = pd.DataFrame(results)

    return df


def build_model():
    """
    Build a simple convolutional network to predict score values. Works on fullsize images (i.e. 1000x1000).
    :return: Keras model for score prediction.
    """

    img_input = keras.Input(shape=(1000, 1000, 1))

    strided = layers.Conv2D(4, 5, strides=(2, 2))(img_input)
    strided = layers.BatchNormalization()(strided)
    strided = layers.LeakyReLU()(strided)

    strided = layers.Conv2D(8, 5, strides=(2, 2))(strided)
    strided = layers.BatchNormalization()(strided)
    strided = layers.LeakyReLU()(strided)

    strided = layers.Conv2D(16, 5, strides=(2, 2))(strided)
    strided = layers.BatchNormalization()(strided)
    strided = layers.LeakyReLU()(strided)

    strided = layers.Conv2D(32, 5, strides=(2, 2))(strided)
    strided = layers.BatchNormalization()(strided)
    strided = layers.LeakyReLU()(strided)

    strided = layers.Conv2D(64, 5, strides=(2, 2))(strided)
    strided = layers.BatchNormalization()(strided)
    strided = layers.LeakyReLU()(strided)

    strided = layers.Conv2D(128, 5, strides=(2, 2))(strided)
    strided = layers.BatchNormalization()(strided)
    strided = layers.LeakyReLU()(strided)

    strided = layers.Conv2D(256, 5, strides=(2, 2))(strided)
    strided = layers.BatchNormalization()(strided)
    strided = layers.LeakyReLU()(strided)

    out = layers.Flatten()(strided)
    out = layers.Dense(1)(out)

    model = keras.Model(inputs=img_input, outputs=out, name='predictor_model')

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', type=int, help='number of epochs to train the model', default=60)
    parser.add_argument('--data_root', type=str, default='/cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/',
                        help='path to the dataset')
    parser.add_argument('--chkpnt_file', type=str, default="simple_cnn.h5", help='Checkpoint file for training.')
    parser.add_argument('--output_dir', type=str, default='./')
    args = parser.parse_args()

    config = DataUtilsConfig()
    config.data_root = args.data_root
    config.batch_size = 32
    config.img_size = 1000
    config.buffer_size = 256
    config.score_threshold = 0.0

    model = build_model()
    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=Adam(lr=0.001), metrics=['mae'])
    model.summary()

    # load dataset
    train_data, val_data, mean, stdev = load_score_prediction_data(config)

    # train
    checkpoint = ModelCheckpoint(args.chkpnt_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=6, factor=0.5, verbose=2)
    csv_logger = CSVLogger(filename=os.path.join(args.output_dir, "simple_cnn_log.csv"), separator=',', append=True)
    callbacks_list = [checkpoint, early, redonplat, csv_logger]
    model.fit(train_data, validation_data=val_data, epochs=args.epochs, verbose=2, callbacks=callbacks_list)

    # predict query:
    model = load_model(args.chkpnt_file)
    query_dataset = load_query_data(config)
    output_df = make_query_prediction(model, query_dataset)

    output_df['Predicted'] *= stdev
    output_df['Predicted'] += mean

    # basic post-processing: we know that all values are from [0; 8]
    preds = list(output_df["Predicted"].values)
    preds = [p if (p >= 0 and p <= 8) else 0 for p in preds]
    output_df["Predicted"] = preds

    output_df.to_csv("finetuned_predictions.csv", index=False)
