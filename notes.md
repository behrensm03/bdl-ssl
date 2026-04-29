### Some things to look into / think about
- tune L2 normalization (weight decay in adam, i believe), and learning rates
    - could be different for different methods because of labeled vs unlabeled set size
- if bcnn can't work out, should we try last layer only?
- Possible issue: I saw rho_init = -5.0 collapse to the majority class in one instance, with the same hyperparameters, and not collapse in another run. The only difference was in the run that collapsed, I forgot to set the random seed. This means, that the initialization randomness can have drastic effects on the end result, which is not ideal. We may have to try everything at multiple random seeds and average, which is annoying.
- try different learning rates for mu and rho. 
- try a two phase approach of only learning mu (with small rho) and then in phase 2 allow rho to update. this might negate or mitigate the importance of rho initialization.
- in soft PL method 20 epochs was probably not enough for the 0.0001 learning rate and that's why all those versions performed worse