import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import warnings
import statistics
import matplotlib.pyplot as plt


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# tf.enable_eager_execution()
# tf.reset_default_graph()
seed = 0
tf.set_random_seed(seed)
random.seed(seed)
np.random.seed(seed)

print("==============================ParseArgs==============================")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--training', action='store_true', default=True)
parser.add_argument('--test_model', default='model_0.8549_logcosh_handcrafted.h5')
args = parser.parse_args()
print(args)

damaged = ['damaged1.csv', 'damaged2.csv', 'damaged3.csv', 'damaged4.csv', 'damaged5.csv']
undamaged = ['undamaged1.csv', 'undamaged2.csv', 'undamaged3.csv', 'undamaged4.csv', 'undamaged5.csv']
# damaged = ['damaged4.csv', 'damaged5.csv']
# undamaged = ['undamaged4.csv', 'undamaged5.csv']

f = lambda x: pd.read_csv(x, header=None, encoding='utf8')
pd_damaged = list(map(f, damaged))
pd_undamaged = list(map(f, undamaged))

print("==============================DataShape==============================")
for i in pd_damaged:
    print(i.shape, end="\t")
print()
for i in pd_undamaged:
    print(i.shape, end="\t")
print()

f = lambda x: x.values
np_damaged = list(map(f, pd_damaged))
np_undamaged = list(map(f, pd_undamaged))

print("==============================Label==============================")
from tensorflow.keras.utils import to_categorical

f = lambda x: to_categorical(np.ones((x.shape[1])), num_classes=2).T
np_y_damaged = list(map(f, pd_damaged))
f = lambda x: to_categorical(np.zeros((x.shape[1])), num_classes=2).T
np_y_undamaged = list(map(f, pd_undamaged))

print(np_damaged[0].shape, type(np_damaged[0]), np_y_damaged[0].shape, type(np_y_damaged[0]))

print("==============================MergeData==============================")
X = np.concatenate(np_damaged + np_undamaged, axis=1).T
Y = np.concatenate(np_y_damaged + np_y_undamaged, axis=1).T
print(X.shape, Y.shape)

print("==============================CalcFeatures==============================")
from scipy.stats import *
from statsmodels.tsa.ar_model import AR

kurt_ = kurtosis(X, axis=1).reshape(-1, 1)
skew_ = skew(X, axis=1).reshape(-1, 1)
var_ = np.var(X, axis=1).reshape(-1, 1)
mean_ = np.mean(X, axis=1).reshape(-1, 1)
# f = lambda x: AR(x).fit().params
# AR_ = np.apply_along_axis(f, 1, X)


f = lambda x: np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] \
                              for i in range(1, 15)])
auto_cor = np.apply_along_axis(f, 1, X)
print("Auto Cor shape", auto_cor.shape)
X_features = np.concatenate([kurt_, skew_, var_, mean_, auto_cor], axis=1)
print("Features shape", X_features.shape)

print("==============================SplitData==============================")
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test, X_features_train, X_features_test = \
    train_test_split(X, Y, X_features, test_size=0.3, random_state=0)
print("Test shape:", x_test.shape, y_test.shape)

if args.training:
    print("==============================Training==============================")
    timesteps = 8192
    n = 19
    sequence = tf.keras.layers.Input(shape=(timesteps, 1), name='Sequence')
    features = tf.keras.layers.Input(shape=(n,), name='Features')

    conv = tf.keras.Sequential()
    conv.add(tf.keras.layers.Conv1D(10, 5, activation='relu', input_shape=(timesteps, 1)))
    conv.add(tf.keras.layers.Conv1D(10, 5, activation='relu'))
    conv.add(tf.keras.layers.MaxPool1D(2))
    conv.add(tf.keras.layers.Dropout(0.5, seed=789))

    conv.add(tf.keras.layers.Conv1D(5, 6, activation='relu'))
    conv.add(tf.keras.layers.Conv1D(5, 6, activation='relu'))
    conv.add(tf.keras.layers.MaxPool1D(2))
    conv.add(tf.keras.layers.Dropout(0.5, seed=789))
    conv.add(tf.keras.layers.Flatten())
    part1 = conv(sequence)

    merged = tf.keras.layers.concatenate([part1, features])

    final = tf.keras.layers.Dense(512, activation='relu')(merged)
    # final = tf.keras.layers.Dense(512, activation='relu')(final)
    final = tf.keras.layers.Dropout(0.5, seed=789)(final)
    final = tf.keras.layers.Dense(2, activation='softmax')(final)

    model = tf.keras.models.Model(inputs=[sequence, features], outputs=[final])
    model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    cp_best = tf.keras.callbacks.ModelCheckpoint(
        "SHM_Conv1_handcrafted.h5", monitor='val_loss', verbose=0,
        save_best_only=True
    )

    # best = 0
    # for i in range(500):
    history = model.fit([np.expand_dims(x_train, 2), X_features_train], y_train,
                        validation_data=([np.expand_dims(x_test, 2), X_features_test], y_test), epochs=500,
                        callbacks=[cp_best], verbose=1)
    # res = model.evaluate([np.expand_dims(x_test, 2), X_features_test], y_test)
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # if res[1] > best:
    #     best = res[1]
    # model.save(f"model_{max(history.history['val_acc']):.4f}_logcosh_handcrafted.h5")
    print("Kết quả valid tot nhat: ", max.mean(history.history['val_acc']))
    print("Kết quả valid trung binh: ", statistics.mean(history.history['val_acc']))

if args.test_model:
    print("==============================Testing==============================")
    # from tensorflow.keras.models import load_model
    #
    # loaded_model = load_model(args.test_model)
    # res = loaded_model.evaluate([np.expand_dims(x_test, 2), X_features_test], y_test)
    # print("Kết quả test: ", res[1])
