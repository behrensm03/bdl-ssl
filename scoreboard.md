# Results of Training Runs

### Hyperparameters, fixed across all runs
- Architecture: 5 layer CNN (see cnn.py)
- Optimizer: Adam
- Epochs: 20 with early stopping, selected by validation AUC
- Batch size: 128
- Loss: cross entropy
- Alpha (weight of unlabeled loss in SSL implementation): 0.5

### Baseline 0: Completely supervised CNN (Upper Bound)
| Val AUC | Test AUC | Date Last Run |
| ------ | ------ | ------ | 
| 0.9202 | 0.9116 | 04/07 |

### Baseline 1: SSL CNN with Hard Pseudo-Labels
| Unlabeled Rate | Threshold | Val AUC | Test AUC | Date Last Run |
| ------- | ------- | -------- | ------- | ----- |
| 50% | 0.95 | 0.9134 | 0.8975 | 04/07 |
| 75% | 0.95 | 0.8853 | 0.8666 | 04/07 |
| 90% | 0.95 | 0.8506 | 0.8325 | 04/07 |


### Baseline 2: Bayesian CNN with Hard Pseudo-Labels
todo

### Method: Bayesian CNN with Soft Pseudo-Labels
todo


## Experiment: Do we need burn-in iterations?
Testing on non bayesian CNN

without burn-in: 
Best Epoch: 9 | Val AUC: 0.8997 | Val Acc: 0.7368 
Test AUC: 0.8893

with 1 burn-in epoch:
Best Epoch: 17 | Val AUC: 0.8978 | Val Acc: 0.7368
Test AUC: 0.8843

with 3 burn-in epochs:
Best Epoch: 8 | Val AUC: 0.9033 | Val Acc: 0.7129
Test AUC: 0.8941

with 5 burn-in epochs:
Best Epoch: 7 | Val AUC: 0.9071 | Val Acc: 0.7049
Test AUC: 0.8792

result: somewhat inconclusive - performance decreases slightly from 0 to 1 burn-in epoch (though only marginal) and then increases marginally at 3 burn-in epochs, then decreases again. could either set it to 3 burn-in epochs or note small difference and abandon burn-in approach.
