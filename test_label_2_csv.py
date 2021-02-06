import numpy as np
import pandas as pd
import glob
import os
import tensorflow as tf
from random import shuffle
import random

FOLDER = "train_validation/"
csv_path = FOLDER + "/label.csv"


def test2csv(tp='', FOLDER="", label_num=3862, csv_path=""):
    tfiles = sorted(glob.glob(os.path.join(FOLDER, tp, '*tfrecord')))
    print('total files in %s %d' % (tp, len(tfiles)))
    with open(csv_path, 'w') as f:
        f.write('VideoId,Label')
        f.write('\n')
        for index_of_tfiles, fn in enumerate(tfiles):
            print("index_of_tfiles", index_of_tfiles)
            for example in tf.compat.v1.python_io.tf_record_iterator(fn):
                tf_example = tf.train.Example.FromString(example)
                ids = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
                yss = np.array(
                    tf_example.features.feature['labels'].int64_list.value)

                each_str = ids + ',' + ' '.join([str(e) for e in yss])

                # print("each_str", each_str)
                f.write(each_str)
                f.write('\n')
    return None


if __name__ == "__main__":
    # test2csv('train_s', label_num=3862, csv_path='train_s_label.csv')
    # test2csv('validation_s', label_num=3862, csv_path='validation_s_label.csv')
    test2csv('train_validation', label_num=3862, csv_path='train_validation_label.csv')
