# Defenses

Analysis of the 2 different strategies that could prevent the attack and eliminate the poisoned samples.

### Agglomerative Clustering Defense
All the test were carried out with 8000 not-poisoned samples and 600 poisoned ones, due to the time and memory required for the program needed to run.

|          | Image space model 3% | Latent space model 3% |
|:--------------:|:-------------:|:-------------:|
| Not-poisoned data identified as poisoned   |          16% |          16% |
| Poisoned data identified as poisoned |          57% |          64% |

### Monitoring Training Dynamics Defense
All the test were carried out with 6000 not-poisoned samples and 600 poisoned ones, due to the time and memory required for the program needed to run.

|          | Image space model 3% | Latent space model 3% |
|:--------------:|:-------------:|:-------------:|
| Not-poisoned data identified as poisoned   |          10% |          9% |
| Poisoned data identified as poisoned |          63% |          66% |


## Evaluation across density functions

| Nuber of interpolations between 4s and 9s | Test Accuracy | Misclassifications |
|:---------:|:-------------:|:---------:|
| **5 interpolations** 	| 92.89% | 9.54% |
| **11 interpolations**	| 92.29% | 8.17% |
| **16 interpolations**	| 92.51% | 8.59% |
| **21 interpolations**	| 92.64% | 10.06% |
| **31 interpolations**	| 92.74% | 9.46% |
| **41 interpolations**	| 91.95% | 8.42% |