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

# perform the AgglomerativeClustering algorithm to identify poisoned samples
def agglomerative_clustering_defense():
    clustering = AgglomerativeClustering(n_clusters = 10).fit(x_poison_dataset)

    # check how many elements have been identified as poisoned
    poisoned_cluster = Counter(clustering.labels_).most_common(1)[0][0]
    # second = Counter(clustering.labels_).most_common(2)[1][0]
    clean_count = 0
    poisoned_count = 0
    for i in range(len(clustering.labels_) - 1, 0, -1):
        if(clustering.labels_[i] == poisoned_cluster):
            if(y_poison_dataset[i] == 0):
                clean_count += 1
            else:
                poisoned_count += 1
    print("Agglomerative Clustering Defense")
    print("Percentage of not-poisoned data identified as poisoned: " + str(clean_count * 100 // (len(y_poison_dataset) - sum(y_poison_dataset))) + "% (" + str(clean_count) + "/" + str(len(y_poison_dataset) - sum(y_poison_dataset)) + ")")
    print("Percentage of poisoned data identified as poisoned: " + str(poisoned_count * 100 // sum(y_poison_dataset)) + "% (" + str(poisoned_count) + "/" + str(sum(y_poison_dataset)) + ")")
# perform the m
def monitoring_training_dynamics_defense(predictons, number_of_poisoned_samples):
    number_of_samples = len(predictons[0])
    number_of_epochs = len(predictons)

    # calculate the difference in the models predictions on a particular example from one epoch to the next
    # ∂fθi(uj) = fθi+1(uj) − fθi(uj)
    # µ(a,b)j = [∂fθa(uj), ∂fθa+1(uj), ... ∂fθb−1(uj), fθb(uj)]
    prediction_changing = [i for i in range(number_of_samples)]
    for example in range(number_of_samples):
        example_changes = []
        for e in range(number_of_epochs - 1):
            example_changes.append(predictons[e + 1][example] - predictons[e][example])
        prediction_changing[example] = example_changes

    # calculate the influence of example i on j
    # Influence(ui,uj) = ||µi(0,K−2) − µj(1,K−1)||2^2
    influence = np.empty((number_of_samples, number_of_samples))
    for i in range(0, number_of_samples):
        for j in range(0, number_of_samples):
            influence[i][j] = np.sum(np.power((np.array(prediction_changing[i][:len(prediction_changing[0]) - 1]) - np.array(prediction_changing[j][1:len(prediction_changing[0])])), 2))
    
    # calculate the average influence of the k nearest neighbors
    # avg influence(u) = 1/k ∑v∈U Influence(u, v)·1[closek(u, v)]
    avg_influence = np.empty(number_of_samples)
    k = 10
    for i in range(number_of_samples):
        close_k = influence[i]
        close_k = sorted(close_k, reverse = True)        
        avg_influence[i] = sum(close_k[:k]) / k

    # check how many elements have been identified as poisoned
    not_poisoned_identified = 0
    poisoned_identified = 0
    threshold = np.percentile(avg_influence, 85)
    for i in range(number_of_samples):
        if(avg_influence[i] >= threshold):
            if(i <= number_of_samples - number_of_poisoned_samples):
                not_poisoned_identified += 1
            else:
                poisoned_identified += 1
    print("Monitoring Training Dynamics Defense")
    print("Percentage of not-poisoned data identified as poisoned: " + str(100 * not_poisoned_identified // (number_of_samples- number_of_poisoned_samples))+ "% (" + str(not_poisoned_identified) + "/" + str(number_of_samples - number_of_poisoned_samples) + ")")
    print("Percentage of poisoned data identified as poisoned: " + str(100 * poisoned_identified // number_of_poisoned_samples) + "% (" + str(poisoned_identified) + "/" + str(number_of_poisoned_samples) + ")")