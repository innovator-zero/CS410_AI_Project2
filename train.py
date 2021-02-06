import tensorflow as tf
import os
import numpy as np
import pandas as pd
from utils_mod import tf_itr, MAP_at_10
from methods import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def train(train_relative_path, val_relative_path, FOLDER):
    weight_folder = 'weights'
    if not os.path.exists(weight_folder): os.mkdir(weight_folder)

    batch = 128
    n_itr = 100
    n_eph = 1
    label_num = 3862
    _, x1_val, x2_val, y_val = next(tf_itr(val_relative_path, 10000, label_num=label_num, FOLDER=FOLDER))
    val_in = np.concatenate((x1_val, x2_val), axis=1)

    cate = pd.read_csv('Category.csv', sep=',')['Label_num'].values  # category distribution

    # LR
    # model = LR(3862)

    # shared-bottom/SB
    # model = SB(cate, [16], [2048, 4096])

    # SNR
    # model = SNR_Trans(cate, [16] ,bottom_0_struct=[512,512,512,512,512,512,512,512],bottom_1_struct=[512,512,512,512,512,512,512,512],beta=0.9, zeta=1.1, gamma=-0.5)

    # CNN
    # model = CNN2(3862)

    # FCN
    model = FCN(3862)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    cnt = 0
    for e in range(n_eph):
        for d in tf_itr(train_relative_path, batch, label_num=label_num, FOLDER=FOLDER):
            _, x1_trn, x2_trn, y_trn = d

            trn_in = np.concatenate((x1_trn, x2_trn), axis=1)
            loss = model.train_on_batch(trn_in, y_trn)
            cnt += 1

            if cnt % n_itr == 0:
                y_prd = model.predict(val_in, verbose=False, batch_size=1000)
                g = MAP_at_10(y_prd, y_val)
                print('loss %.5f val GAP %0.5f; epoch: %d; iters: %d' % (loss, g, e, cnt))
                model.save_weights(weight_folder + '/%0.5f_%d_%d.h5' % (g, e, cnt))


if __name__ == '__main__':
    train_relative_path = 'train'
    val_relative_path = 'validation'
    FOLDER = ''
    train(train_relative_path, val_relative_path, FOLDER)
