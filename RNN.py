
import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten#, CuDNNLSTM

# Load the dataset ...
#  You will need to seperately download or generate this file
Xd = pickle.load( open( "RML2016.10a_dict.pkl", "rb" ), encoding='latin' )
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
print(mods)
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.7)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

in_shp = list(X_train.shape[1:])

classes = mods #The categories

#Building the rnn model
model = Sequential()

#For Cpu
# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
model.add(LSTM(256, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(80, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(classes), activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# #For GPU
# # IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
# model.add(CuDNNLSTM(256, input_shape=(X_train.shape[1:]), return_sequences=True))
# model.add(Dropout(0.5))
#
# model.add(CuDNNLSTM(80))
# model.add(Dropout(0.5))
#
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(len(classes), activation='softmax'))
#
# opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy'],
)
model.summary()
#
model.fit(X_train,
          Y_train,
          epochs=3,
          validation_data=(X_test, Y_test))

