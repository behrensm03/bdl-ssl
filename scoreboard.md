# Results of Training Runs

### Hyperparameters, fixed across all runs
- Architecture: 5 layer CNN (see cnn.py)
- Optimizer: Adam, lr=0.001
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