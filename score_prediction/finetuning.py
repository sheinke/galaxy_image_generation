import os
import argparse
import random
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

import sys
sys.path.append("..")

from utils.data_utils import DataUtilsConfig, load_query_data, load_img, img_subtract_mean, \
    img_binarise, mirror, rotate, random_crop

'''
This script takes a pretrained discriminator (e.g. from a WGAN or a StyleGAN) and
freezes all its layers. It then adds new layers at a specified point in the old model
which are trained on score prediction. Lastly, the new (i.e. extended) model is
unfrozen and all layers are jointly trained.
'''


def do_rebalancing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance the data according to the score distribution, by assigning individual
    augmentation factors.
    :param df: Input dataframe, containing IDs and scores.

    :return: Balanced dataframe.
    """
    scores = list(df['Actual'].values)
    num_buckets = 8 * 4  # scores interval is [0; 8], stepsize of 0.25
    bucket_counts = [0] * num_buckets
    for s in scores:
        bucket_counts[int(s * 4)] += 1

    # still want to augment the peak of the distribution by some factor
    max_count = max(bucket_counts) * 4
    df_output = []
    for index, row in df.iterrows():
        augment_factor = max_count / bucket_counts[int(row['Actual'] * 4)]
        # we want every sample at least once:
        augment_factor = max(augment_factor, 1)
        # have to account for the fact that the augment function
        # only delivers a certain number of augmented samples:
        # change 16 to the correct value, if you change the data config !
        augment_factor = min(augment_factor, 16)
        # approximate fraction with random uniform dist:
        residual = (augment_factor * 100) % 100
        if random.randint(0, 100) <= residual:
            augment_factor += 1
        augment_factor = int(augment_factor)
        df_output.append({'Id': int(row['Id']),
                          'Actual': row['Actual'],
                          'aug_factor': augment_factor})

    df_output = pd.DataFrame(df_output)

    return df_output


def augment_img_by_factor(img, score,  augment_factor, img_size=1000, pad_size=100, do_random_crop=True, nr_shifts=2):
    """
    Augments the given sample. Image dimensions are kept.
    :param img: Input image.
    :param score: Score value for given input image.
    :param augment_factor: Determines the number of augmented samples (per individual "raw" sample)
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

    random.shuffle(result)

    augment_factor = tf.reduce_max([1, augment_factor])
    augment_factor = tf.reduce_min([len(result), augment_factor])

    aug_imgs = tf.data.Dataset.from_tensor_slices(result)
    img_idxs = tf.data.Dataset.range(0, len(result))
    temp = tf.data.Dataset.zip((aug_imgs, img_idxs))
    temp = temp.filter(lambda m_img, counter: tf.cast(counter, tf.int32) < tf.cast(augment_factor, tf.int32))
    result = temp.map(lambda m_img, counter: (m_img, score))

    return result


def load_score_prediction_data(config, val_split=0.1, rebalance=False):
    """
    Loads a dataset that can be used for score prediction. This function is highly similar to
    data_utils.load_data, but it performs val-splitting as well as can perform a rough balancing of the
    score distribution. Since we do not always need that for our generative models, we decided
    to factor this functionality out into this function.
    :param config: DataUtilsConfig object
    :return: tf.data.Dataset with augmented images and scores
    """
    print("Loading score prediction data from: {}".format(config.data_root))
    scored_file_path = os.path.join(config.data_root, "scored.csv")
    scored_input_dir = os.path.join(config.data_root, "scored/")
    # get scores and scores filepaths
    scores = pd.read_csv(scored_file_path)
    # split into train and val set
    scores_train, scores_val, _, _ = train_test_split(scores, scores, test_size=val_split)

    if rebalance:
        # calculate optimal augment factors for train set:
        scores_train = do_rebalancing(scores_train)

    # normalize: if rebalancing is not done, than we can simply take the usual statistics,
    # since the augment factor is equal for all samples
    if not rebalance:
        mean, stdev = np.mean(scores_train['Actual']), np.std(scores_train['Actual'])
    else:
        scores_repeated_by_aug_factor = []
        for index, row in scores_train.iterrows():
            scores_repeated_by_aug_factor += [row['Actual']] * int(row['aug_factor'])

        mean, stdev = np.mean(scores_repeated_by_aug_factor), np.std(scores_repeated_by_aug_factor)

    scores_train['Actual'] = scores_train['Actual'].map(lambda x: (x - mean) / stdev)
    scores_val['Actual'] = scores_val['Actual'].map(lambda x: (x - mean) / stdev)

    # get filepaths and score values as lists:
    scores_filepaths_train = [scored_input_dir + str(id_) + '.png' for id_ in scores_train['Id'].to_numpy()]
    scores_filepaths_val = [scored_input_dir + str(id_) + '.png' for id_ in scores_val['Id'].to_numpy()]

    # convert to TF datasets and join together
    ds_name_train = tf.data.Dataset.from_tensor_slices(np.array(scores_filepaths_train))
    ds_scores_train = tf.data.Dataset.from_tensor_slices(scores_train['Actual'].to_numpy())
    if rebalance:
        ds_augfacs_train = tf.data.Dataset.from_tensor_slices(scores_train['aug_factor'].to_numpy())
    else:
        # 42000 is arbitrary, has to exceed maximum augment factor that the respective data util config produces
        ds_augfacs_train = tf.data.Dataset.from_tensor_slices(np.array([42000] * len(scores_train)))
    ds_train = tf.data.Dataset.zip((ds_name_train, ds_scores_train, ds_augfacs_train))

    ds_name_val = tf.data.Dataset.from_tensor_slices(np.array(scores_filepaths_val))
    ds_scores_val = tf.data.Dataset.from_tensor_slices(scores_val['Actual'].to_numpy())
    ds_val = tf.data.Dataset.zip((ds_name_val, ds_scores_val))

    # filter out based on score threshold
    ds_train = ds_train.filter(lambda name, score, aug_factor: score >= (config.score_threshold - mean) / stdev)
    # for validation set filter nonsense images
    ds_val = ds_val.filter(lambda name, score: score >= (config.score_threshold - mean) / stdev)

    # load images
    ds_train = ds_train.map(lambda name, score, aug_factor: (load_img(name, config.img_size, config.resize_method),
                                                             score,
                                                             aug_factor),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(lambda name, score: (load_img(name, config.img_size, config.resize_method), score),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #  augment train set
    ds_train = ds_train.flat_map(lambda img, score, aug_factor: augment_img_by_factor(img=img,
                                                                                      score=score,
                                                                                      augment_factor=aug_factor,
                                                                                      img_size=config.img_size,
                                                                                      pad_size=config.pad_size,
                                                                                      do_random_crop=config.do_random_crop,
                                                                                      nr_shifts=config.nr_shifts))

    # subtract image mean
    if config.img_subtract_mean:
        ds_train = ds_train.map(lambda image, score: (img_subtract_mean(image), score))
        ds_val = ds_val.map(lambda image, score: (img_subtract_mean(image), score))

    # binarise images
    elif config.img_binarise:  # else-if so we do not execute img_subtract_mean twice
        ds_train = ds_train.map(lambda image, score: (img_binarise(image), score))
        ds_val = ds_val.map(lambda image, score: (img_binarise(image), score))

    # shuffle and batch
    ds_train = ds_train.shuffle(config.buffer_size).batch(config.batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.shuffle(config.buffer_size).batch(config.batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # hyperparameters for architecture and training
    parser.add_argument('--epochs', type=int, help='number of epochs to train the model', default=1000)
    parser.add_argument('--batch_size', type=int, default=16, help='# images per batch')
    parser.add_argument('--img_size', type=int, default=1000, help='image width')
    # input- / output-directories
    parser.add_argument('--data_root', type=str,
                        default='/cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/',
                        help='path to the dataset')
    parser.add_argument('--model_file', type=str, default="wgan_trained_discr.h5", help='H5 file for model')
    parser.add_argument('--transfer_layer', type=str, default="concatenate_2",
                        help='the name of the layer in the old model, where we start adding new layers')
    parser.add_argument('--chkpnt_file', type=str, default="tuned_predictor",
                        help='Checkpoint name for training (no file-ending).')
    parser.add_argument('--output_dir', type=str, default='./')
    args = parser.parse_args()

    # CAUTION, if you change the data config, you may need to adjust the upper bound
    # for the augmentation factor during rebalancing ! See function do_rebalancing for more details.
    config = DataUtilsConfig()
    config.data_root = args.data_root
    config.batch_size = args.batch_size
    config.img_size = args.img_size
    config.buffer_size = 256
    config.score_threshold = 0.0


    # load frozen pretrained model
    old_model = load_model(args.model_file)
    # freeze the complete old model
    for layer in old_model.layers:
        layer.trainable = False

    # add new layers: have to assign layer names explicitly, since default names are already taken
    new_conv = Conv2D(256, 3, name="new_conv1")(old_model.get_layer(args.transfer_layer).output)
    new_conv = BatchNormalization(name="new_bn1")(new_conv)
    new_conv = LeakyReLU(name="new_leaky1")(new_conv)

    new_conv = Conv2D(512, 3, name="new_conv2")(new_conv)
    new_conv = BatchNormalization(name="new_bn2")(new_conv)
    new_conv = LeakyReLU(name="new_leaky2")(new_conv)

    new_dense = Flatten(name="new_flatten1")(new_conv)
    new_dense = Dense(512, name="new_dense1")(new_dense)
    new_dense = BatchNormalization(name="new_bn3")(new_dense)
    new_dense = LeakyReLU(name="new_leaky3")(new_dense)

    new_dense = Dense(1024, name="new_dense2")(new_dense)
    new_dense = BatchNormalization(name="new_bn4")(new_dense)
    new_dense = LeakyReLU(name="new_leaky4")(new_dense)

    new_out = Dense(1, name="new_dense3", activation="linear")(new_dense)

    model = Model(inputs=old_model.input, outputs=new_out)
    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=Adam(lr=0.002), metrics=['mae'])
    model.summary()

    # load dataset
    train_data, val_data, mean, stdev = load_score_prediction_data(config)
    print("Mean score of train data: ", mean)
    print("Stdev of train data scores: ", stdev)

    # train model with Keras API: define callback list
    frozen_checkpoint = args.chkpnt_file + "_frozen_baselayers.h5"
    checkpoint = ModelCheckpoint(frozen_checkpoint, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=6, factor=0.5, verbose=2)
    csv_logger = CSVLogger(filename=os.path.join(args.output_dir, "frozen_log.csv"), separator=',', append=True)
    callbacks_list = [checkpoint, early, redonplat, csv_logger]
    model.fit(train_data, validation_data=val_data, epochs=args.epochs, verbose=2, callbacks=callbacks_list)

    # load best weights and unfreeze *whole* model
    model = load_model(frozen_checkpoint)
    for layer in model.layers:
        layer.trainable = True

    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=Adam(lr=0.0003), metrics=['mae'])
    model.summary()

    # same story, different checkpoint file:
    unfrozen_checkpoint = args.chkpnt_file + "_unfrozen.h5"
    checkpoint = ModelCheckpoint(unfrozen_checkpoint, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="loss", mode="min", patience=10, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="loss", mode="min", patience=6, factor=0.8, verbose=2)
    csv_logger = CSVLogger(filename=os.path.join(args.output_dir, "unfrozen_log.csv"), separator=',', append=True)
    callbacks_list = [checkpoint, early, redonplat, csv_logger]
    model.fit(train_data, validation_data=val_data, epochs=args.epochs, verbose=2, callbacks=callbacks_list)

    # make query prediction:
    model = load_model(unfrozen_checkpoint)

    query_dataset = load_query_data(config)
    output_df = make_query_prediction(model, query_dataset)

    output_df['Predicted'] *= stdev
    output_df['Predicted'] += mean

    # basic post-processing: we know that all values are from [0; 8]
    preds = list(output_df["Predicted"].values)
    preds = [p if (p >= 0 and p <= 8) else 0 for p in preds]
    output_df["Predicted"] = preds

    output_df.to_csv("finetuned_predictions.csv", index=False)