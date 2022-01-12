# Poisoning unlabeled Dataset for Semi Supervised Learning

Project  work  done  on  the  ["Poisoning  the  Unlabeled  Dataset  of Semi-Supervised Learning"](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-poisoning) paper.

This is an imlementatio of a Semi-supervised learning model that recognizes the digits in the MNIST database and of various attacks.

## Project structure
The project has 2 folders:

- `confusion_matrices`, which contains the confusion matrix of each attack run;
- `poisoned_data`, which contains the poisoned data used for the attacks;

The code is separated in 3  files:

- `interpolation.py`, which creates the poisoned dataset;
- `ladder_net.py`, which defines the Semi-supervised learning model;
- `mnist_example.py`, which trains and poison the model;

## Sources
For the poisoning data generation: https://github.com/kvsnoufal/Latent-Space-Interpolation
For the ladder network: https://github.com/divamgupta/ladder_network_keras