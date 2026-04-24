## Experiment: Estimate quality of uncertainty across methods
The goal of this was to answer an open question, that being: 
When we initialize rho to something small, like -5.0, because larger values of rho (such as -2.25) cause the model to collapse to the majority class, are we actually gaining anything by using a BCNN, or is such a value as initial rho small enough that we have effectively created a more computationally expensive CNN?

From what I can tell, the initialization of rho does play a significant role in model convergence. A natural question to ask is, if the initialization of rho is so important, why isn't training pushing it towards the ideal value? Does that mean our KL divergence is not strong enough to push the distribution parameters to where they should be?

I don't know the answer for certain, but all the evidence i have seen so far does not support the idea that our KL weighting is the problem. As a diagnostic, I ran the following sub-experiment.

#### Sub-experiment: tuning the beta parameter
I initialized two networks with the same hyperparameters, but initialized one with `rho_init=-2.25` and the other with `rho_init=-5.0`. I then performed a sweep over values of `beta`, the weight assigned to our normalized KL divergence term.
The finding from this experiment was that regardless of beta, the initialization of rho was what determined the final outcome: the model with `rho_init=-2.25` converged to a degenerate solution that predicted the majority class at least 95% of the time on all inputs, regardless of beta. the model with `rho_init=-5.0` produced "reasonable" predictions for various settings of beta. (reasonable being the correct class generally had higher probability than other classes). All of the beta sweeping was done without unlabeled examples, to eliminate the pseudo labels as a source of error.

It is also worth noting that extreme values of beta (100000) did cause the model's parameters to converge to the prior, so we confirmed there is not an issue in our gradients from this loss term - the model *is* being pushed slightly towards the prior.

To me, this indicated that beta was not the issue, and that `rho_init` was important regardless of how much we weight the KL divergence.

On to the main experiment.

### Experiment

So the goal is to determine, is `rho_init=-5.0` useless and just approximating a basic CNN plus some meaningless noise in each input parameter?

To determine this, we look at some metrics for evaluating our uncertainty quality - the NLL macro-averaged across classes, and the per-class NLL, in addition to our standard metric macro-AUC, all on the test split.

I trained 3 models: baseline CNN, baseline BCNN with `rho_init=-2.25`, and baseline BCNN with `rho_init=-5.0`, and compared those metrics, keeping all other hyperparameters the same, and using the fully labeled dataset (learning rate = 0.001, beta = 10.0).

| Model | rho_init | Test mAUC | Test macro NLL | Per Class NLL |
| ------ | ----- | ------ | ----- | ------ |
| CNN | N/A | 0.919 | 1.649 | 1.510023   1.6546735  1.3569949  3.2449975  1.085825   0.31857845 2.3700705 |
| BCNN | -2.25 | 0.831 | 2.264 | 2.5728407 1.8770232 1.8149303 3.5847878 1.9233601 0.3105795 3.7669992 |
| BCNN | -5.0 | 0.913 | 1.504 | 1.9737276  1.4791938  1.3086464  2.6548927  1.0812091  0.39036497 1.6417654 |

In the BCNN with `rho_init=-2.25` the model collapses to majority-only predictions and this is reflected in worse metrics across the board than other models, except for NLL on majority class (class index 5). 

If we compare the baseline CNN with the BCNN at `rho_init=-5.0`, we see similar mAUC performance, but lower macro NLL on the BCNN. This indicates that the BCNN is indeed adding some value in terms of uncertainty quality, even at a seemingly low `rho_init`. This is evidence that perhaps -5 is simply a reasonable enough initialization for rho, and that it does not completely negate the benefits of a BCNN compared to the CNN.

Looking at the per-class NLL, the BCNN has lower (better) NLL on almost every class, except the majority class and class index 0. Perhaps the majority class is slightly worse because the BCNN is spreading probability mass more evenly. So we basically are trading a tiny amount of mAUC performance and calibration on the majority class for pretty significant looking improvements on other classes, particularly the minority classes (index 3, from 3.24 to 2.65, and index 6 from 2.37 to 1.64).

**To me, this supports the idea that `rho_init=-5.0` is still worth doing because it does improve calibration over the CNN.**

It may just be that the higher rho is too high, and that we find ourselves in some other local minimum during the training process.