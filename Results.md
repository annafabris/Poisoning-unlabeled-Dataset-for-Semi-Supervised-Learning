## Defenses

Analysis of the 2 different strategies that could prevent the attack and eliminate the poisoned samples.

### Agglomerative Clustering Defense
All the test were carried out with 8000 not-poisoned samples and 600 poisoned ones.

|     Method     | Image space model 3% | Latent space model 3% |
|:--------------:|:-------------:|:-------------:|
| Not-poisoned data identified as poisoned   |          16% |          16% |
| Poisoned data identified as poisoned |          57% |          64% |

### Monitoring Training Dynamics Defense
All the test were carried out with 6000 not-poisoned samples and 600 poisoned ones.

|     Method     | Image space model 3% | Latent space model 3% |
|:--------------:|:-------------:|:-------------:|
| Not-poisoned data identified as poisoned   |          10% |          9% |
| Poisoned data identified as poisoned |          63% |          66% |