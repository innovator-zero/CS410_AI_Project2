import numpy as np
from multiprocessing import Pool
import glob
import os
import tensorflow as tf
import pandas as pd


def ap_at_10(data):
    # based on https://github.com/google/youtube-8m/blob/master/average_precision_calculator.py
    predictions_idx, actuals_idx = data
    predictions_idx = np.char.split(predictions_idx).item()
    actuals_idx = np.char.split(actuals_idx).item()
    n = 10
    total_num_positives = None

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be 'None' or a positive integer."
                             " It was '%s'." % n)
    ap = 0.0
    sortidx = predictions_idx  # np.argsort(predictions)[::-1]

    if total_num_positives is None:
        numpos = len(actuals_idx)  # np.size(np.where(actuals > 0))
    else:
        numpos = total_num_positives

    if numpos == 0:
        return 0

    if n is not None:  # in label prediction, the num should be 1 or 0
        numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
        r = min(r, n)
    for i in range(r):
        if sortidx[i] in actuals_idx:
            poscount += 1
            ap += poscount / (i + 1) * delta_recall

    return ap


# restrict on @K is in ap_at_n
def MAP_at_10(pred, actual):
    lst = zip(list(pred), list(actual))

    all = list(map(ap_at_10, lst))
    return np.mean(all)


def test_score(pred, data_type):
    label = data_type + '_label.csv'
    actuals = pd.read_csv(label, sep=",")
    actuals.sort_values("VideoId", inplace=True)
    actuals = actuals["Label"].values
    prediction_10 = pd.read_csv(pred, sep=',')
    prediction_10.sort_values("VideoId", inplace=True)
    prediction_10 = prediction_10["Label"].values

    MAP = MAP_at_10(prediction_10, actuals)
    print(data_type, "MAP %.4f" % (MAP))


test_score('output_518021910066.csv', 'train_validation')
