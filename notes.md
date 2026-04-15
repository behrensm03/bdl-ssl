### Some things to look into / think about
- How do we measure uncertainty quality?
    - Negative log likelihood? Expected Calibration Error?
- Do we actually need burn-in iterations like we proposed? I think i forgot to implement that part of it and we're just directly using the pseudo labels from iteration 1
    - on one hand, we initially thought that the first few training iterations, the pseudo labels produced would basically be garbage and just hurt training. the idea then being, we should use only labeled examples for some small number of burn-in epochs to get a reasonable network before we start letting the unlabeled examples flow through. 
    - on the other hand, images are small here, and dataset isn't that big (especially for some classes), so i think overfitting is clearly a real concern (see plots of models overfitting after ~5 epochs). especially when the unlabeled rate is high, if we only train on labeled examples, we might very quickly (1 epoch or less) overfit to those few training examples. thus, by the time we start introducing unlabeled examples we've already gotten to a place where we're overfit and need to now correct that. could unlabeled examples help correct that? maybe. or maybe not. maybe, eliminating this step entirely and just immediately using all examples can help with overfitting. first of all, for the hard pseudo labels we might not even use most examples right away since they need to meet a threshold, so any unconfident estimates won't be used anyway. but what about for our method? if we're always using them, would that play more of a role?
- Should we normalize loss terms by the number of labeled / unlabeled examples?
    - motivation: suppose we have 5000 unlabeled examples and 500 labeled examples. even though alpha tunes them, with a small alpha we could still have unlabeled loss dominate that term such that the labeled loss isn't contributing much.
- should we look at out of distribution images / see confidence on those?



- tune L2 normalization (weight decay in adam, i believe), and learning rates
    - could be different for different methods because of labeled vs unlabeled set size
- evaluate different thresholds (it would be nice to say which sort of thresholds the soft does better on if any)
- try a quick experiment comparing some small warmup vs not warming up and see what happens