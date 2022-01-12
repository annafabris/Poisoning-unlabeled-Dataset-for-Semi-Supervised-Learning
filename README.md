# Poisoning unlabeled Dataset for Semi Supervised Learning

Project  work  done  on  the  ["Poisoning  the  Unlabeled  Dataset  of Semi-Supervised Learning"](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-poisoning) paper.

This is an imlementation of a Semi-supervised learning model (Ladder Network) that recognizes the digits in the MNIST database.
Following a few attack were executed with the target of misclassifing 4s with 9s.
### Results
|           | Test Accuracy | 9-4 error | 4s classified as 9s |
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
For the poisoning data generation: https://github.com/kvsnoufal/Latent-Space-Interpolation

For the ladder network: https://github.com/divamgupta/ladder_network_keras