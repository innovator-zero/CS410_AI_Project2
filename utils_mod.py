import numpy as np
from multiprocessing import Pool
import glob
import os
import pandas as pd
import tensorflow as tf


def ap_at_10(data):
    # based on https://github.com/google/youtube-8m/blob/master/average_precision_calculator.py
    predictions, actuals = data
    n = 10
    total_num_positives = None

    if len(predictions) != len(actuals):
        raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be 'None' or a positive integer."
                             " It was '%s'." % n)

    ap = 0.0

    sortidx = np.argsort(predictions)[::-1]

    if total_num_positives is None:
        numpos = np.size(np.where(actuals > 0))
    else:
        numpos = total_num_positives

    if numpos == 0:
        return 0

    if n is not None:
        numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
        r = min(r, n)
    for i in range(r):
        if actuals[sortidx[i]] > 0:
            poscount += 1
            ap += poscount / (i + 1) * delta_recall
    return ap

# MAP
# restrict on @K is in ap_at_n
def MAP_at_10(pred, actual):
    lst = zip(list(pred), list(actual))

    with Pool() as pool:
        all = pool.map(ap_at_10, lst)

    return np.mean(all)

def trans_in(label,file):
    new_label=file[label]
    return new_label


# data load generator
def tf_itr(tp='test', batch=1024, label_num=3862, FOLDER="", num=None):
    file = pd.read_csv('Mapping_in.csv', sep=',')['NewLabel'].values
    tfiles = sorted(glob.glob(os.path.join(FOLDER, tp, '*tfrecord')))

    ids, aud, rgb, lbs = [], [], [], []
    if num == None:
        it = len(tfiles)
    else:
        it = num
    for index_i in range(it):
        fn = tfiles[index_i]

        print("\rLoading files: [{0:50s}] {1:.1f}% ".format('#' * int((index_i + 1) / it * 50),
                                                             (index_i + 1) / it * 100), fn, end="", flush=True)
        for example in tf.python_io.tf_record_iterator(fn):
            tf_example = tf.train.Example.FromString(example)
            ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
            rgb.append(np.array(tf_example.features.feature['mean_rgb'].float_list.value).astype(float))
            aud.append(np.array(tf_example.features.feature['mean_audio'].float_list.value).astype(float))

            yss = np.array(tf_example.features.feature['labels'].int64_list.value)
            yss_new=trans_in(yss,file)
            out = np.zeros(label_num).astype(np.int8)
            for y in yss_new:
                out[y] = 1
            lbs.append(out)
            if len(ids) >= batch:
                yield np.array(ids), np.array(aud), np.array(rgb), np.array(lbs)
                ids, aud, rgb, lbs = [], [], [], []
        if index_i + 1 == it:
            yield np.array(ids), np.array(aud), np.array(rgb), np.array(lbs)
            ids, aud, rgb, lbs = [], [], [], []

