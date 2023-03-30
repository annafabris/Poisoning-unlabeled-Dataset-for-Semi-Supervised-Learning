# Poisoning unlabeled Dataset for Semi Supervised Learning

This project is based on the  ["Poisoning  the  Unlabeled  Dataset  of Semi-Supervised Learning"](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-poisoning) paper.

This is a Semi-supervised learning model (Ladder Network) that recognizes the digits in the MNIST database. The aim of the project is to execute attacks to misclassify 4s as 9s.


### Results
The project compares the accuracy of various models, including supervised, non-poisoned, and different types of poisoned models. The table below shows the test accuracy and the percentage of misclassified 4s as 9s and vice versa for each model:

|           | Test Accuracy | 9s misclassified as 4s | 4s misclassified as 9s |
|:---------:|:-------------:|:---------:|---------------------|
| **Supervised model**	 	| 98.88% | 0.4% | 0.5% |
| **Non-poisoned model** 	| 95.46% | 2.3% | 2.7% |
| **Latent space model 3%**	| 92.90% | 0.9% | 9.3% |
| **Image space model 3%**	| 89.04% | 11.8% | 47.6%|
| **Image space model 1%**	| 94.21% | 1.7% | 3.1% |

## Project structure
The project has 2 folders:

- `confusion_matrices`, which contains the confusion matrix of each attack run;
- `poisoned_data`, which contains the poisoned data used for the attacks;

The code is separated in 3  files:

- `interpolation.py`, which creates the poisoned dataset;
- `ladder_net.py`, which defines the Semi-supervised learning model;
- `mnist_example.py`, which trains and poison the model;

## Sources
- https://blog.keras.io/building-autoencoders-in-keras.html
- https://github.com/divamgupta/ladder_network_keras
