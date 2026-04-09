### Some things to look into / think about
- How do we measure uncertainty quality?
    - Negative log likelihood? Expected Calibration Error? Predictive entropy? BALD?
- Do we actually need burn-in iterations like we proposed? I think i forgot to implement that part of it and we're just directly using the pseudo labels from iteration 1
- Should we normalize loss terms by the number of labeled / unlabeled examples?
    - motivation: suppose we have 5000 unlabeled examples and 500 labeled examples. even though alpha tunes them, with a small alpha we could still have unlabeled loss dominate that term such that the labeled loss isn't contributing much.