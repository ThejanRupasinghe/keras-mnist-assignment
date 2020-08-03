# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 20:38:58 2020

@author: Thejan Rupasinghe
"""

# library imports
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Input
from keras.activations import relu, sigmoid
from keras import utils, datasets, Model

# load dataset - test and train
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# convert label (dependent variable) to binary one-hot - to use softmax in NN
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# add noise to train and test images
noise_factor = 0.25
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# convert to float32 and scale variables to 0-1 range - 255 is the max value for a grey scale pixel
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train_noisy = x_train_noisy.astype("float32") / 255
x_test_noisy = x_test_noisy.astype("float32") / 255

# images (28,28) by initial and 1 channel as grey scale 2D, Therefore (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
x_train_noisy = np.expand_dims(x_train_noisy, -1)
x_test_noisy = np.expand_dims(x_test_noisy, -1)

print("x_train Shape:", x_train.shape)
print("Training data:", x_train.shape[0])
print("Testing data:", x_test.shape[0])

# buliding the autoencoder 
autoencoder_nn = Sequential()
autoencoder_nn.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1), padding='same'))
autoencoder_nn.add(MaxPooling2D(pool_size=(2, 2)))
autoencoder_nn.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'))
autoencoder_nn.add(MaxPooling2D(pool_size=(2, 2)))
autoencoder_nn.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'))
autoencoder_nn.add(UpSampling2D(size=(2, 2)))
autoencoder_nn.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'))
autoencoder_nn.add(UpSampling2D(size=(2, 2)))
autoencoder_nn.add(Conv2D(1, kernel_size=(3, 3), activation="sigmoid", padding='same'))
# compile and train
autoencoder_nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
autoencoder_nn.fit(x_train_noisy, x_train, batch_size=256, epochs=3, validation_split=0.1)

# predicting noise removed data from noisy test and train data
x_train_noise_removed = autoencoder_nn.predict(x_train_noisy)
x_test_noise_removed = autoencoder_nn.predict(x_test_noisy)

# buliding the actual number prediction model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

# print summary  
model.summary()

# compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the model
model.fit(x_train_noise_removed, y_train, batch_size=128, epochs=15, validation_split=0.1)

# test the model
score = model.evaluate(x_test_noise_removed, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])