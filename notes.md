### Some things to look into / think about
- if bcnn can't work out, should we try last layer only?
- try different learning rates for mu and rho. 
- try a two phase approach of only learning mu (with small rho) and then in phase 2 allow rho to update. this might negate or mitigate the importance of rho initialization.
- in soft PL method 20 epochs was probably not enough for the 0.0001 learning rate and that's why all those versions performed worse
- we weren't really consistent with how we did hyperparam tuning between CNN / BCNN / Soft PL