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
