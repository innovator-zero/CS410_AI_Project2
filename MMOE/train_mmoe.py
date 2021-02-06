# inspired by https://github.com/drawbridge/keras-mmoe

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from utils_mod import tf_itr, MAP_at_10
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input, Dense, Concatenate
from keras.initializers import VarianceScaling
from keras.models import Model
# from FCNmodel import build_model
from mmoe import MMoE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def train(train_relative_path, val_relative_path, FOLDER):
    weight_folder = 'weights'
    if not os.path.exists(weight_folder): os.mkdir(weight_folder)
    batch = 128
    n_itr = 100
    n_eph = 15
    label_num = 3862
    _, x1_val, x2_val, y_val = next(tf_itr(val_relative_path, 10000, label_num=label_num, FOLDER=FOLDER))
    val_in = np.concatenate((x1_val, x2_val), axis=1)

    cate = pd.read_csv('Category.csv', sep=',')['Label_num'].values

    # Set up the input layer
    input_layer = Input(shape=(1152,))

    # Set up MMoE layer
    mmoe_layers = MMoE(
        units=128,
        num_experts=8,
        num_tasks=25
    )(input_layer)

    output_layers = []

    #Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=16,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=cate[index],
            activation='sigmoid',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)


    out = Concatenate(axis=1)(output_layers)

    # Compile model
    model = Model(inputs=input_layer, outputs=out)
    adam_optimizer = Adam(lr=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam_optimizer
    )

    # Print out model architecture summary
    model.summary()

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
