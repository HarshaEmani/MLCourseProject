# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:34:33 2024

@author: harsh
"""

from tensorflow.keras import datasets
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, InputLayer, Conv1D, Flatten, GRU, RepeatVector, MaxPooling1D
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_pinball_loss
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.activations import swish
from statsmodels.graphics.tsaplots import plot_acf
import datetime
import random
import math
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def print_metrics(y_true, y_pred):
    print("=================================================")
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))
    print("Precision Score: ", precision_score(y_true, y_pred, average='micro'))
    print("Recall Score: ", recall_score(y_true, y_pred, average='micro'))
    print("F1 Score: ", f1_score(y_true, y_pred, average='micro'))
    print("=================================================")


(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

X_train.shape, y_train.shape, X_test.shape, y_test.shape

model = Sequential()
model.add(InputLayer(shape=(28, 28)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
# opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=35, batch_size=64)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

results = pd.DataFrame({'Actual': y_test, 'Predictions': y_pred})

print_metrics(y_test, y_pred)

notMatchIndex = y_pred != y_test

test = results[notMatchIndex]
# train_results = X_test[notMatchIndex]
