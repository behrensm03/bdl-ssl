# Results of Training Runs

### Hyperparameters, fixed across all runs
- Architecture: 5 layer CNN (see cnn.py)
- Optimizer: Adam
- Epochs: 20 with early stopping, selected by validation AUC
- Batch size: 128
- Loss: cross entropy
- Alpha (weight of unlabeled loss in SSL implementation): 0.5

### Baseline 0: Completely supervised CNN (Upper Bound)
| Val mAUC | Test mAUC | Test gAUC | Date Last Run |
| ------ | ------ | ----- | ------ | 
| 0.9202 | 0.9116 | ____ | 04/07 |
| 0.9254 | 0.9253 | 0.9637 | 04/17 |

### Baseline 1: SSL CNN with Hard Pseudo-Labels - Hyperparam tuning
#### 50% Unlabeled Rate
| Threshold | learning rate | L2 strength | Val mAUC | Val gAUC | Date Last Run |
| ------- | ------- | ------- | -------- | ------- | ----- |
| 0.95 | 0.0001 | 0.001 | 0.9139 | 0.9567 | 04/17 |
| 0.95 | 0.001 | 0.0001 | 0.9135 | 0.9541 | 04/17 |
| 0.95 | 0.0001 | 0.0001 | 0.9128 | 0.9576 | 04/17 |
| 0.95 | 0.0001 | 0.01 | 0.9120 | 0.9566 | 04/17 |
| 0.95 | 0.001 | 0.0 | 0.9084 | 0.9566 | 04/17 |
| 0.95 | 0.001 | 0.001 | 0.9058 | 0.9566 | 04/17 |
| 0.95 | 0.0001 | 0.0 | 0.9038 | 0.9555 | 04/17 |
| 0.95 | 0.001 | 0.01 | 0.9002 | 0.9565 | 04/17 |
| 0.95 | 0.01 | 0.0001 | 0.8917 | 0.9478 | 04/17 |
| 0.95 | 0.01 | 0.0 | 0.8806 | 0.9445 | 04/17 |
| 0.95 | 0.01 | 0.001 | 0.8649 | 0.9424 | 04/17 |
| 0.95 | 0.01 | 0.01 | 0.8546 | 0.9395 | 04/17 |

#### Best setting: lr = 0.0001, L2 = 0.001
| Val mAUC | Val gAUC | Test mAUC |
| ------ | ------ | ------ |


#### 75% Unlabeled Rate
| Threshold | learning rate | L2 strength | Val mAUC | Val gAUC | Date Last Run |
| ------- | ------- | ------- | -------- | ------- | ----- |
| 0.95 | 0.0001 | 0.01 | 0.9086 | 0.9527 | 04/17 |
| 0.95 | 0.0001 | 0.0001 | 0.9067 | 0.9507 | 04/17 |
| 0.95 | 0.0001 | 0.0 | 0.9054 | 0.9493 | 04/17 |
| 0.95 | 0.001 | 0.0 | 0.9034 | 0.9528 | 04/17 |
| 0.95 | 0.0001 | 0.001 | 0.8970 | 0.9424 | 04/17 |
| 0.95 | 0.001 | 0.001 | 0.8953 | 0.9510 | 04/17 |
| 0.95 | 0.001 | 0.0001 | 0.8952 | 0.9472 | 04/17 |
| 0.95 | 0.001 | 0.01 | 0.8893 | 0.9477 | 04/17 |
| 0.95 | 0.01 | 0.0 | 0.8663 | 0.9413 | 04/17 |
| 0.95 | 0.01 | 0.0001 | 0.8572 | 0.9422 | 04/17 |
| 0.95 | 0.01 | 0.001 | 0.8510 | 0.9393 | 04/17 |
| 0.95 | 0.01 | 0.01 | 0.7907 | 0.9128 | 04/17 |

#### Best setting: 
| Val mAUC | Val gAUC | Test mAUC |
| ------ | ------ | ------ |

#### 90% Unlabeled Rate
| Threshold | learning rate | L2 strength | Val mAUC | Val gAUC | Date Last Run |
| ------- | ------- | ------- | -------- | ------- | ----- |
| 0.95 | 0.0001 | 0.001 | 0.8838 | 0.9437 | 04/17 |
| 0.95 | 0.0001 | 0.01 | 0.8747 | 0.9442 | 04/17 |
| 0.95 | 0.0001 | 0.0001 | 0.8657 | 0.9412 | 04/17 |
| 0.95 | 0.001 | 0.001 | 0.8626 | 0.9411 | 04/17 |
| 0.95 | 0.0001 | 0.0 | 0.8618 | 0.9393 | 04/17 |
| 0.95 | 0.001 | 0.01 | 0.8606 | 0.9411 | 04/17 |
| 0.95 | 0.001 | 0.0001 | 0.8531 | 0.9350 | 04/17 |
| 0.95 | 0.001 | 0.0 | 0.8488 | 0.9376 | 04/17 |
| 0.95 | 0.01 | 0.0001 | 0.8391 | 0.9357 | 04/17 |
| 0.95 | 0.01 | 0.0 | 0.8140 | 0.9311 | 04/17 |
| 0.95 | 0.01 | 0.001 | 0.7907 | 0.9241 | 04/17 |
| 0.95 | 0.01 | 0.01 | 0.7863 | 0.9159 | 04/17 |


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
