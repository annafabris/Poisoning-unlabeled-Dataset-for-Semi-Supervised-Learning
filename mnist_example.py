from __future__ import print_function
from ssl import OP_NO_RENEGOTIATION

import numpy as np
import keras
import random
import csv
import os

from sklearn.metrics import accuracy_score
from keras.datasets import mnist

from ladder_net import get_ladder_network_fc
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#get the dataset
inp_size = 28 * 28 # size of mnist dataset 
n_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, inp_size).astype('float32')/255
x_test  = x_test.reshape(10000,  inp_size).astype('float32')/255

y_train = keras.utils.np_utils.to_categorical(y_train, n_classes)
y_test  = keras.utils.np_utils.to_categorical(y_test,  n_classes)

# only select 100 training samples to keep the labels
idxs_annot = range(x_train.shape[0])
random.seed(0)
idxs_annot = np.random.choice(x_train.shape[0], 100)

x_train_unlabeled = x_train
x_train_labeled   = x_train[idxs_annot]
y_train_labeled   = y_train[idxs_annot]

# choose which pre donwloaded data to include
interpolation = 'image'
if(interpolation == 'image'):      # 1800 images of 4-9 image space interpolation
    path = os.path.dirname(os.path.abspath(__file__)) + '/poisoned_data/image_space_1800.csv'
elif(interpolation == 'latent'):    # 1800 images of 4-9 latent space interpolation
    path = os.path.dirname(os.path.abspath(__file__)) + '/poisoned_data/latent_space_1800.csv'

# choose which percentage of the poisoned dataset to add
percentage = 3      # 3%
poisoned_data_size = x_train.shape[0] // 100 * percentage

# initialize the model 
model = get_ladder_network_fc(layer_sizes=[inp_size, 1000, 500, 250, 250, 250, n_classes])

# download poisoned data and add it
#with open(path, newline='') as f:
#    reader = csv.reader(f)
#    x_train_poisoned = list(reader)
#
#print(poisoned_data_size)
#x_train_poisoned = np.concatenate((x_train_unlabeled, x_train_poisoned[1:poisoned_data_size + 1]))
#x_train_poisoned = x_train_poisoned.astype(float)

n_rep = x_train_unlabeled.shape[0] // x_train_labeled.shape[0]
x_train_labeled_rep = np.concatenate([x_train_labeled]*n_rep)
y_train_labeled_rep = np.concatenate([y_train_labeled]*n_rep)

# initialize the model 
model = get_ladder_network_fc(layer_sizes=[inp_size, 1000, 500, 250, 250, 250, n_classes])

# train the model for 10 epochs
for _ in range(10):
    model.fit([x_train_labeled_rep, x_train_unlabeled], y_train_labeled_rep, epochs=1)
    y_test_pr = model.test_model.predict(x_test, batch_size=100)
    print("Test accuracy : %f" % accuracy_score(y_test.argmax(-1), y_test_pr.argmax(-1)))

# display the confusion matrix
display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test.argmax(-1), y_test_pr.argmax(-1)))
fig, ax = plt.subplots(figsize=(20, 15))
display.plot(include_values=True, ax=ax)
plt.show()
plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/confusion_matrices/' + interpolation + '_space_' + str(poisoned_data_size))