import keras
import random
from keras.datasets import mnist
from keras.datasets import mnist
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import csv
import os
from collections import Counter

#get the dataset
inp_size = 28 * 28 # size of mnist dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, inp_size).astype('float32')/255.0
y_train =[1 for i in x_train]

# choose which pre donwloaded data to include
interpolation = 'image'
if(interpolation == 'image'):      # 1800 images of 4-9 image space interpolation
    path = os.path.dirname(os.path.abspath(__file__)) + '/poisoned_data/image_space_1800.csv'
elif(interpolation == 'latent'):    # 1800 images of 4-9 latent space interpolation
    path = os.path.dirname(os.path.abspath(__file__)) + '/poisoned_data/latent_space_1800.csv'

# download poisoned data and add it
with open(path, newline='') as f:
    reader = csv.reader(f)
    poison_dataset = np.array(list(reader))

x_poison_dataset = np.concatenate((x_train, poison_dataset[1:]))
x_poison_dataset = x_poison_dataset.astype(float)
y_poison_dataset = [0 if i < len(x_train) else 1 for i in range(len(x_poison_dataset))]

# perform the AgglomerativeClustering algorithm
clustering = AgglomerativeClustering(n_clusters = 10).fit(x_poison_dataset)

# check how many elements have been identified as poisoned
poisoned_cluster = Counter(clustering.labels_).most_common(1)[0][0]
second = Counter(clustering.labels_).most_common(2)[1][0]
clean_count = 0
poisoned_count = 0
for i in range(len(clustering.labels_) - 1, 0, -1):
    if(clustering.labels_[i] == poisoned_cluster or clustering.labels_[i] == second):
        if(y_poison_dataset[i] == 0):
            clean_count += 1
        else:
            poisoned_count += 1
print("Percentage of not-poisoned data identified as poisoned: " + str(100 - (len(y_poison_dataset) - clean_count) * 100 // len(y_poison_dataset)) + "% (" + str(clean_count) + "/" + str(len(y_poison_dataset)) + ")")
print("Percentage of poisoned data identified as poisoned: " + str(100 - 100*(sum(y_poison_dataset) - poisoned_count) // sum(y_poison_dataset)) + "% (" + str(poisoned_count) + "/" + str(sum(y_poison_dataset)) + ")")