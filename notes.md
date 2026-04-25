### Some things to look into / think about
- Should we normalize loss terms by the number of labeled / unlabeled examples?
    - motivation: suppose we have 5000 unlabeled examples and 500 labeled examples. even though alpha tunes them, with a small alpha we could still have unlabeled loss dominate that term such that the labeled loss isn't contributing much.
- should we look at out of distribution images / see confidence on those?
- tune L2 normalization (weight decay in adam, i believe), and learning rates
    - could be different for different methods because of labeled vs unlabeled set size
- evaluate different thresholds (it would be nice to say which sort of thresholds the soft does better on if any)
- should we scale / reweight the KL loss term? seems like it might dominate
- worth noting: we strip labels for training but not val/test and that might be kind of an unrealistic (but reasonable) thing. so like our 7000 training examples might become 700 labeled / 6300 unlabeled via the 90% unlabeled rate. but then all of a sudden we evaluate on 1000 labeled examples in validation and 2000 labeled examples in test? i think it's fine because we need labeled examples to evaluate on, but it maybe is a bit unrealistic that we technically have 3700 labeled examples in this case, but are only using 700 in training. if the dataset were truly SSL out of the box, we probably would not use so many examples on the val/test splits.

- if bcnn can't work out, should we try last layer only?
- Possible issue: I saw rho_init = -5.0 collapse to the majority class in one instance, with the same hyperparameters, and not collapse in another run. The only difference was in the run that collapsed, I forgot to set the random seed. This means, that the initialization randomness can have drastic effects on the end result, which is not ideal. We may have to try everything at multiple random seeds and average, which is annoying.
- try different learning rates for mu and rho. 
- try a two phase approach of only learning mu (with small rho) and then in phase 2 allow rho to update. this might negate or mitigate the importance of rho initialization.
