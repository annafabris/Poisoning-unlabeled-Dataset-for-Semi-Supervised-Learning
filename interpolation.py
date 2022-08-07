import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import csv
import os

from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.datasets import mnist
from pathlib import Path

# choose which percentage of the poisoned dataset to add
percentage = 3  # 3%
# choose wheter to generare latent space interpolation or image space interpolation
latent = True  # True if latent space interpolation, false otherwise
# chose the details of the interpolation
num_images_per_interpolation = 25
images_to_remove = (
    2  # remove the first and last few images from the interpolation as they're trivial
)
start_interpolation_number = 4
end_interpolation_number = 9


# split train set into train and validation
def train_val_split(x_train, y_train):
    rnd = np.random.RandomState(seed=42)
    perm = rnd.permutation(len(x_train))
    train_idx = perm[: int(0.8 * len(x_train))]
    val_idx = perm[int(0.8 * len(x_train)) :]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]


# generate interpolation
def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


# Add gaussian, poisson and speckle noise to an image
def noise(noise_type, image):
    if noise_type == "gauss":
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var**1.8
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        row, col = image.shape
        gauss = np.random.randn(row, col) * 0.05
        gauss = gauss.reshape(row, col)
        noisy = image + image * gauss
        return noisy


# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255
y_train = keras.utils.np_utils.to_categorical(y_train, 10)
y_test = keras.utils.np_utils.to_categorical(y_test, 10)

# split train set
n = 20000
index = np.random.choice(x_train.shape[0], n)
x_train = x_train[index]
x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)
max_value = float(x_train.max())
x_train = x_train.astype("float32") / max_value
x_val = x_val.astype("float32") / max_value

# "encoded" is the encoded representation of the input
input = Input(shape=(x_train.shape[1:]))
encoded = Conv2D(16, (3, 3), activation="relu", padding="same")(input)
encoded = MaxPooling2D((2, 2), padding="same")(encoded)
encoded = Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
encoded = MaxPooling2D((2, 2), padding="same")(encoded)
encoded = Conv2D(8, (3, 3), strides=(2, 2), activation="relu", padding="same")(encoded)
encoded = Flatten()(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Reshape((4, 4, 8))(encoded)
decoded = Conv2D(8, (3, 3), activation="relu", padding="same")(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(8, (3, 3), activation="relu", padding="same")(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(16, (3, 3), activation="relu")(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(decoded)

# Create the autoencoder model
autoencoder = Model(input, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# Create the encoder model
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[6].output)

# Create the decoder model
encoded_input = Input(shape=(128,))
decoder_layer = autoencoder.layers[-8](encoded_input)
decoder_layer = autoencoder.layers[-7](decoder_layer)
decoder_layer = autoencoder.layers[-6](decoder_layer)
decoder_layer = autoencoder.layers[-5](decoder_layer)
decoder_layer = autoencoder.layers[-4](decoder_layer)
decoder_layer = autoencoder.layers[-3](decoder_layer)
decoder_layer = autoencoder.layers[-2](decoder_layer)
decoder_layer = autoencoder.layers[-1](decoder_layer)
decoder = Model(encoded_input, decoder_layer)

# train model
history = autoencoder.fit(x_train, x_train, epochs=100, validation_data=(x_val, x_val))

# get data ready for interpolation
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255

interpolated_images = []
start_interpolation = []
end_interpolation = []
interpolation = []

# calculate the percentage of the poisoned dataset to add
poisoned_data_size = x_train.shape[0] // 100 * percentage
num_interpolations = poisoned_data_size // (num_images_per_interpolation - 4)

if latent:
    # get start_interpolation_number and end_interpolation_number
    for index, el in enumerate(y_train[0 : (poisoned_data_size + 1) * 20]):
        if el == start_interpolation_number:
            start_interpolation.append(
                encoder.predict(x_train[index].reshape(1, 28, 28, 1))
            )
        if el == end_interpolation_number:
            end_interpolation.append(
                encoder.predict(x_train[index].reshape(1, 28, 28, 1))
            )
    # compute the interpolations
    for o, t in zip(
        start_interpolation[0:num_interpolations],
        end_interpolation[0:num_interpolations],
    ):
        interpolated_images.append(
            interpolate_points(o, t, num_images_per_interpolation)
        )
    for j in interpolated_images:
        for i in range(images_to_remove, len(j) - images_to_remove):
            interpolation.append(decoder.predict(j[i].reshape(1, 128)).reshape(28, 28))
else:
    # get start_interpolation_number and end_interpolation_number
    for index, el in enumerate(y_train[0 : (poisoned_data_size + 1) * 20]):
        if el == start_interpolation_number:
            start_interpolation.append((x_train[index].reshape(1, 28, 28, 1)))
        if el == end_interpolation_number:
            end_interpolation.append((x_train[index].reshape(1, 28, 28, 1)))
    # compute the interpolations
    for o, t in zip(
        start_interpolation[0:num_interpolations],
        end_interpolation[0:num_interpolations],
    ):
        interpolated_images.append(
            interpolate_points(o, t, num_images_per_interpolation)
        )
    for j in interpolated_images:
        for i in range(images_to_remove, len(j) - images_to_remove):
            interpolation.append((j[i]).reshape(28, 28))

# compose the path of the poisoned data
directory = str(Path().absolute())
if latent:
    path = (directory+ "/poisoned_data/"+ "latent_space_"+ str(poisoned_data_size)+ ".csv"
    )
else:
    path = (
        directory
        + "/poisoned_data/"
        + "image_space_"
        + str(poisoned_data_size)
        + ".csv"
    )

# save poisoned dataset
noise_type = "gauss"
poisoned_dataset = []
for i in interpolation[:poisoned_data_size]:
    if noise_type != "none":
        poisoned_dataset.append(noise(i).flatten())
    else:
        poisoned_dataset.append(i.flatten())
df = pd.DataFrame(poisoned_dataset)
df.to_csv(path, index=False)
