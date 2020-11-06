
import os
from os import listdir
from os.path import isfile, join
import argparse
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

import sys
sys.path.append("..")
from utils.data_utils import load_img

'''
This script is supposed to make score prediction as simple as possible, in order to make comparisons between
different generative models easier. It reads a number of PNG images from a directory and feeds them into the
specified predictor model. The biggest drawback is that it requires manual entry of the mean score value and
the standard deviation of the dataset that the predictor model was trained on (since it needs to denormalize
the model outputs to make them interpretable).
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_file', type=str, default='tuned_predictor_unfrozen.h5', help='H5 file for model')
    parser.add_argument('--img_size', type=int, default=1000, help='image width')
    parser.add_argument('--resize_method', type=str, default='bilinear', help='{bilinear, padded}')
    parser.add_argument('--img_dir', type=str, default='./', help='path to directory with PNG images')
    parser.add_argument('--out_csv', type=str, default='judge_results.csv', help='output file for result CSV file.')
    parser.add_argument('--score_pred_mean', type=float, default=1.6525513001952545, help='Mean score value for denormalization.')
    parser.add_argument('--score_pred_stdev', type=float, default=1.0817796577706593, help='Standard deviation of score values for denormalization.')
    args = parser.parse_args()

    # load predictor
    predictor = load_model(args.model_file)

    # load images 
    imgs_files = [f for f in listdir(args.img_dir) if isfile(join(args.img_dir, f)) and f.endswith("png")]
    # cut off the ".png", take the remaining file names as IDs
    imgs_id_strings = [str(f[:-4]) for f in imgs_files]
    # fully extend file names
    imgs_files = [join(args.img_dir, f) for f in imgs_files]
    # convert into TF datasets:
    ds_name = tf.data.Dataset.from_tensor_slices(np.array(imgs_files))
    ds_ids = tf.data.Dataset.from_tensor_slices(np.array(imgs_id_strings))

    ds_picture = ds_name.map(lambda x: load_img(x, args.img_size, args.resize_method),
    						 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = tf.data.Dataset.zip((ds_picture, ds_ids))
    # batch size of 8 is arbitrary, only concern here is not to exceed GPU memory.
    ds = ds.batch(8).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # start prediction:
    results = []  # list of dictionaries
    for image_batch, ids_batch in ds:
        prediction = predictor(image_batch, training=False)
        for i in range(prediction.shape[0]):  # assume channels-last format
            results.append({"Id": ids_batch.numpy()[i].decode('UTF-8'), "Predicted": prediction.numpy()[i, 0]})          
    
    df = pd.DataFrame(results)

    # denormalize
    df['Predicted'] *= args.score_pred_stdev
    df['Predicted'] += args.score_pred_mean

    # basic post-processing: we know that all values are from [0; 8]
    preds = list(df["Predicted"].values)
    preds = [p if (p >= 0 and p <= 8) else 0 for p in preds]
    df["Predicted"] = preds

    # write to output file:
    df.to_csv(args.out_csv, index=False)
    print("Score Statistics:")
    print("MIN: ", np.min(df['Predicted']))
    print("MAX: ", np.max(df['Predicted']))
    print("AVG: ", np.mean(df['Predicted']))
    print("STD: ", np.std(df['Predicted']))
