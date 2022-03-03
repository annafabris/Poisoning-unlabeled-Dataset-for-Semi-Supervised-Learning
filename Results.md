# Defenses

Analysis of the 2 different strategies that could prevent the attack and eliminate the poisoned samples.

### Agglomerative Clustering Defense
All the test were carried out with 8000 not-poisoned samples and 600 poisoned ones, due to the time and memory required for the program needed to run.

|          | Image space model | Latent space model |
|:--------------:|:-------------:|:-------------:|
| Not-poisoned data identified as poisoned   |          16% |          16% |
| Poisoned data identified as poisoned |          57% |          64% |

### Monitoring Training Dynamics Defense
All the test were carried out with 6000 not-poisoned samples and 600 poisoned ones, due to the time and memory required for the program needed to run.

|          | Image space model | Latent space model |
|:--------------:|:-------------:|:-------------:|
| Not-poisoned data identified as poisoned   |          10% |          9% |
| Poisoned data identified as poisoned |          63% |          66% |

# Evaluation across density functions
All the test were carried out with 3% of poisoned sample on the image space interpolation.

| Number of interpolations between 4s and 9s | Test Accuracy | Misclassifications |
|:---------:|:-------------:|:---------:|
| **5 interpolations** 	| 92.89% | 9.54% |
| **11 interpolations**	| 92.29% | 8.17% |
| **16 interpolations**	| 92.51% | 8.59% |
| **21 interpolations**	| 92.64% | 10.06% |
| **31 interpolations**	| 92.74% | 9.46% |
| **41 interpolations**	| 91.95% | 8.42% |


# Evaluation across numbers of supervised label
All the test were carried out on the image space interpolation.

| Number of labels | Test Accuracy | Misclassifications |
|:---------:|:-------------:|:---------:|
| **100** 	| 92.64% | 10.06% |
| **1000**	| 96.36% | 0.01% |

# Attacks with adding noise to the poisoned examples
All the test were carried out with 3% of poisoned sample on the image space interpolation.

|          | Test Accuracy | Misclassifications |
|:--------------:|:-------------:|:-------------:|
| No Noise   |          92.64% |          10.06% |
| Gaussian Noise   |          90.72% |          13.01% |
| Poisson Noise |          93.35% |          8.20% |
| Speckle Noise |          91.42% |          13.62% |
