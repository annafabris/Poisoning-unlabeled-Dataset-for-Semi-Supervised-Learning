import keras
import random
from keras.datasets import mnist
import os

#get the dataset
inp_size = 28 * 28 # size of mnist dataset 
n_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, inp_size).astype('float32')/255
x_test  = x_test.reshape(10000,  inp_size).astype('float32')/255

y_train = keras.utils.np_utils.to_categorical(y_train, n_classes)
y_test  = keras.utils.np_utils.to_categorical(y_test,  n_classes)

# choose which pre donwloaded data to include
interpolation = 'image'
if(interpolation == 'image'):      # 1800 images of 4-9 image space interpolation
    path = os.path.dirname(os.path.abspath(__file__)) + '/poisoned_data/image_space_1800.csv'
elif(interpolation == 'latent'):    # 1800 images of 4-9 latent space interpolation
    path = os.path.dirname(os.path.abspath(__file__)) + '/poisoned_data/latent_space_1800.csv'
