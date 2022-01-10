import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import csv
import os

from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input, Reshape ,Flatten, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.datasets import mnist

# split train set into train and validation
def train_val_split(x_train, y_train):
    rnd = np.random.RandomState(seed = 42)
    perm = rnd.permutation(len(x_train))
    train_idx = perm[:int(0.8 * len(x_train))]
    val_idx = perm[int(0.8 * len(x_train)):]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]

# generate interpolation
def interpolate_points(p1, p2, n_steps = 10):
		# interpolate ratios between the points
		ratios = np.linspace(0, 1, num = n_steps)
		# linear interpolate vectors
		vectors = list()
		for ratio in ratios:
				v = (1.0 - ratio) * p1 + ratio * p2
				vectors.append(v)
		return np.asarray(vectors)

# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test  = x_test.reshape(10000,  28, 28, 1).astype('float32')/255
y_train = keras.utils.np_utils.to_categorical(y_train, 10)
y_test  = keras.utils.np_utils.to_categorical(y_test,  10)

# split train set
n = 20000  
index = np.random.choice(x_train.shape[0], n) 
x_train = x_train[index]
x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_val = x_val.astype('float32') / max_value

# define models
input = Input(shape = (x_train.shape[1:]))
encoded = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(input)
encoded = MaxPooling2D((2, 2), padding = 'same')(encoded)
encoded = Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(encoded)
encoded = MaxPooling2D((2, 2), padding = 'same')(encoded)
encoded = Conv2D(8, (3, 3), strides = (2,2), activation = 'relu', padding = 'same')(encoded)
encoded = Flatten()(encoded)

decoded = Reshape((4, 4, 8))(encoded)
decoded = Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(16, (3, 3), activation = 'relu')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(1, (3, 3), activation = 'sigmoid', padding = 'same')(decoded)

autoencoder = Model(input,decoded)

encoder = Model(inputs = autoencoder.input, outputs = autoencoder.layers[6].output)
encoded_input = Input(shape = (128,))

deco = autoencoder.layers[-8](encoded_input)
deco = autoencoder.layers[-7](deco)
deco = autoencoder.layers[-6](deco)
deco = autoencoder.layers[-5](deco)
deco = autoencoder.layers[-4](deco)
deco = autoencoder.layers[-3](deco)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)

decoder = Model(encoded_input, deco)

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

gen = image.ImageDataGenerator()
batches = gen.flow(x_train, x_train, batch_size = 128)
val_batches = gen.flow(x_val, x_val, batch_size = 128)

# train model
history = autoencoder.fit_generator(generator = batches, epochs = 100, 
                    validation_data = val_batches, validation_steps = val_batches.n)

# get data ready for interpolation
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255

latent = True # True if latent space interpolation, false otherwise
num_images_per_interpolation = 25
num_interpolations = 9
start_interpolation_number = 4
end_interpolation_number = 9
interpolated_images = []
start_interpolation = []
end_interpolation = []
interpolation = []

if (latent):
    # get start_interpolation_number and end_interpolation_number
    for index, el in enumerate(y_train[0:num_interpolations * 10]):
        if el == start_interpolation_number:
            start_interpolation.append(encoder.predict(x_train[index].reshape(1,28,28,1)))
        if el == end_interpolation_number:
            end_interpolation.append(encoder.predict(x_train[index].reshape(1,28,28,1)))
    # compute the interpolations
    for o, t in zip(start_interpolation[0:num_interpolations], end_interpolation[0:num_interpolations]):
        interpolated_images.append(interpolate_points(o, t, num_images_per_interpolation))
    for j in interpolated_images:
        for i in range(2, len(j) - 2):
            interpolation.append(decoder.predict(j[i].reshape(1,128)).reshape(28,28))
else:
    # get start_interpolation_number and end_interpolation_number
    for index, el in enumerate(y_train[0:num_interpolations * 10]):
        if el == start_interpolation_number:
            start_interpolation.append((x_train[index].reshape(1,28,28,1)))
        if el == end_interpolation_number:
            end_interpolation.append((x_train[index].reshape(1,28,28,1)))
    # compute the interpolations
    for o, t in zip(start_interpolation[0:num_interpolations], end_interpolation[0:num_interpolations]):
        interpolated_images.append(interpolate_points(o, t, num_images_per_interpolation))
    for j in interpolated_images:
        for i in range(2, len(j) - 2):
            interpolation.append((j[i]).reshape(28,28))

# choose which percentage of the poisoned dataset to add
percentage = 3      # 3%
poisoned_data_size = x_train.shape[0] // 100 * percentage

# choose which pre donwloaded data to include
if(latent):
    path = os.path.dirname(os.path.abspath(__file__) + '/poisoned_data' + 'latent_space_' + str(poisoned_data_size) + '.csv')
else:
    path = os.path.dirname(os.path.abspath(__file__) + '/poisoned_data' + 'image_space_' + str(poisoned_data_size) + '.csv')

# save poisoned dataset
poisoned_dataset = []
for i in interpolation[:poisoned_data_size]:
    poisoned_dataset.append(i.flatten())
df = pd.DataFrame(poisoned_dataset) 
df.to_csv(path, index=False)