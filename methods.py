import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Multiply, Softmax, concatenate, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class LR(keras.Model):

    def __init__(self, categories):
        super().__init__()
        self.lr = Dense(categories, activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.lr(inputs)
        return x


class SB(keras.Model):

    def __init__(self, categories: list, tower_struct: list, bottom_struct: list):
        super().__init__()

        self.bottom = keras.models.Sequential(name='bottom')
        for size in bottom_struct:
            self.bottom.add(Dense(size))
            self.bottom.add(LeakyReLU())

        self.num_towers = len(categories)
        self.towers = []

        for i in range(self.num_towers):
            num_class = categories[i]
            tower = keras.models.Sequential(name='tower_%s' % i)

            for size in tower_struct:
                tower.add(Dense(size, activation='relu'))
            tower.add(Dense(num_class, activation='sigmoid'))

            self.towers.append(tower)

        self.bn = BatchNormalization()
        self.dropout = Dropout(0.2)

    def call(self, inputs, training=None):
        x = inputs

        x = self.bottom(inputs)
        x = self.bn(x)
        x = self.dropout(x)

        tower_out = []
        for tower in self.towers:
            tower_out.append(tower(x))
            print(tower_out)
        x = tf.concat(tower_out, 1)

        return x


class SNR_Trans(keras.Model):

    def __init__(self, categories: list, tower_struct: list, bottom_0_struct: list, bottom_1_struct: list,
                 beta, zeta, gamma):
        super().__init__()

        self.num_expert_0 = len(bottom_0_struct)
        self.num_expert_1 = len(bottom_1_struct)
        self.num_towers = len(categories)
        self.z_num = self.num_expert_0 * self.num_expert_1 + self.num_expert_1 * self.num_towers

        self.alpha = self.add_weight(name='alpha',
                                     shape=[self.z_num],
                                     initializer='normal',
                                     trainable=True)

        self.bottom_0 = []
        self.bottom_1 = []
        self.towers = []

        for num in bottom_0_struct:
            layer = keras.models.Sequential()
            layer.add(Dense(num, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
            self.bottom_0.append(layer)

        for num in bottom_1_struct:
            layer = keras.models.Sequential()
            layer.add(Dense(num, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
            layer.add(BatchNormalization())
            layer.add(Dropout(0.2))
            self.bottom_1.append(layer)

        for i in range(self.num_towers):
            num_class = categories[i]
            tower = keras.models.Sequential(name='tower_%s' % i)

            for size in tower_struct:
                tower.add(Dense(size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
            tower.add(Dense(num_class, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))

            self.towers.append(tower)

        self.beta, self.zeta, self.gamma = beta, zeta, gamma

    def call(self, inputs, training=None):
        x = inputs

        z_cnt = 0

        U = tf.random_uniform([self.z_num], minval=0, maxval=1)
        S = K.sigmoid((K.log(U) - K.log(1 - U) + self.alpha) / self.beta)
        S = S * (self.zeta - self.gamma) + self.gamma
        zs = S

        out_b0 = []
        for layer in self.bottom_0:
            out_b0.append(layer(x))

        out_b1 = []
        for layer in self.bottom_1:
            x = tf.concat([zs[z_cnt + i] * out_b0[i] for i in range(self.num_expert_0)], 1)
            z_cnt += self.num_expert_0
            out_b1.append(layer(x))

        tower_out = []
        for tower in self.towers:
            x = tf.concat([zs[z_cnt + i] * out_b1[i] for i in range(self.num_expert_1)], 1)
            z_cnt += self.num_expert_1
            tower_out.append(tower(x))
        x = tf.concat(tower_out, 1)

        return x


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *


def CNN(size):
    model = Sequential()
    model.add(Reshape((1024 + 128, 1)))
    model.add(Conv1D(8, 3, activation='relu'))
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(size, activation='sigmoid'))

    return model


def CNN2(size):
    model = Sequential()
    model.add(Reshape((1024 + 128, 1)))
    model.add(Conv1D(32, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(size, activation='sigmoid'))

    return model


def FCN(size):
    model = Sequential()
    model.add(Dense(8192, activation='relu'))
    model.add(Dropout(0.02))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(size, activation='sigmoid'))
    
    return model
