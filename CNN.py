import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation, Add, AveragePooling2D, ZeroPadding2D, Reshape

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
n_train = int(n_examples * 0.5)
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

# Build VT-CNN2 Neural Net model using Keras primitives --
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

dr = 0.5 # dropout rate (%)
model = Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(256, 1, strides = 3, padding='valid', activation="relu", name="conv1", kernel_initializer = "glorot_uniform",
                 data_format='channels_first'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(80, 2, strides=3, padding="valid", activation="relu", name="conv2", kernel_initializer = "glorot_uniform"))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer ='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense( len(classes), kernel_initializer ='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Set up some params
nb_epoch = 100     # number of epochs to train on
batch_size = 1024  # training batch size

model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          verbose=2,
          validation_data=(X_test, Y_test),
          )

# model.fit(X_train,
#           Y_train,
#           epochs=3,
#           validation_data=(X_test, Y_test))