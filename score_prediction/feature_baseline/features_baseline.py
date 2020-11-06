import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

def parse_csv(data_file):
    """
    Parses csv output of Mathematica script.
    :param data_file: path to csv
    :return: parsed data frame used for training
    """
    features = pd.read_csv(data_file, delimiter=";", keep_default_na=False, na_values=['Indeterminate']).T
    features.reset_index(level=0, inplace=True)
    features = features.rename({"index": "Id"}, axis=1).astype({"Id": "int64"}).dropna()
    return features


def train_and_validate(wavelet, data_root):
    """
    Train and computes MAE error on the validation set.
    :param wavelet: haar or coiflet (see Mathematica script)
    :param data_root: data directory
    """
    if wavelet == "haar":
        features = parse_csv("HaarFeatures.csv")
    elif wavelet == "coiflet":
        features = parse_csv("CoifletFeatures.csv")
    else:
        print("error: {} not supported".format(wavelet))

    scores = pd.read_csv(os.path.join(data_root, "scored.csv")).astype({"Id": "int64"})

    data = features.merge(scores, on="Id", how="inner")
    nr_features = 31
    X = data.iloc[:, 1:(nr_features + 1)]
    Y = data.iloc[:, (nr_features + 1)]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.1)
    regr = HistGradientBoostingRegressor(loss='least_absolute_deviation', max_iter=500, max_leaf_nodes=100)
    regr.fit(X_train, y_train)
    error = np.mean(np.abs(regr.predict(X_test) - y_test))
    plt.scatter(regr.predict(X_test), y_test)
    print("validation MAE: {}".format(error))
    plt.show()


def train_and_prepare_submisssion(wavelet, data_root):
    """
    Trains on the whole dataset and creates predictions for the query data.
    :param wavelet: haar or coiflet (see Mathematica script)
    :param data_root: data directory
    """
    if wavelet == "haar":
        features = parse_csv("HaarFeatures.csv")
        test_features = parse_csv("HaarFeatures_query.csv")
    elif wavelet == "coiflet":
        features = parse_csv("CoifletFeatures.csv")
        test_features = parse_csv("CoifletFeatures_query.csv")
    else:
        print("error: {} not supported".format(wavelet))

    scores = pd.read_csv(os.path.join(data_root, "scored.csv")).astype({"Id": "int64"})

    data = features.merge(scores, on="Id", how="inner")
    nr_features = 31
    X = data.iloc[:, 1:(nr_features + 1)]
    Y = data.iloc[:, (nr_features + 1)]

    X_test = test_features.iloc[:, 1:(nr_features + 1)]

    X_train, y_train, = X, Y
    regr = HistGradientBoostingRegressor(loss='least_absolute_deviation', max_iter=500, max_leaf_nodes=100)

    regr.fit(X_train, y_train)
    predictions = regr.predict(X_test)
    submission = pd.DataFrame({"Id": test_features["Id"], "Predicted": predictions}).set_index("Id")
    submission.to_csv("features_baseline_predictions.csv")


def create_mean_baseline_submission(data_root):
    """
    Creates submission with score average.
    :param data_root: data directory
    """
    scores = pd.read_csv(os.path.join(data_root, "scored.csv")).astype({"Id": "int64"})
    mean = scores["Actual"].mean()
    query_dir = os.path.join(data_root, "query/")
    query_pics_files = [f for f in os.listdir(query_dir) if os.path.isfile(os.path.join(query_dir, f)) and f.endswith("png")]
    query_pics_ids = [int(f[:-4]) for f in query_pics_files]
    predictions = [mean] * len(query_pics_ids)
    submission = pd.DataFrame({"Id": query_pics_ids, "Predicted": predictions}).set_index("Id")
    submission.to_csv("mean_predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_mean_score', type=int, help='0 if wavelet features, 1 scores mean', default=0)
    parser.add_argument('--data_root', type=str, default='/cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/',
                        help='path to the dataset')

    args = parser.parse_args()
    if not args.output_mean_score:
        train_and_prepare_submisssion("coiflet", args.data_root)  # use coiflet because of better validation scores
    else:
        create_mean_baseline_submission(args.data_root)
