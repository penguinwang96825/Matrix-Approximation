# Matrix Factorisation Project

We propose a novel de-mixing scheme which works by singular value decomposition (SVD) decorrelating in embedding domain. With the help of SVD properties, this approach is better than performing de-mixing functions to extract the target speakers' embeddings. Compared to de-mixing functions, the presented approach doesn't require pre-computed speaker inventory containing profile vector (d-vector) of the speaker. Instead of utilising SVD for the purposes of signal reconstruction or data compression, the proposed approach explains statistical variation among attributes for building an orthonormal embedding space. This interpretation is what makes it useful in the context of signal processing. Up-to-date repository can be found from [here](https://github.com/asifjalal/speaker-embedding-factorisation).

# Quick Experiment

## Prepare

Run `sh prepare/prepare_timit.sh` in order to start the TIMIT experiment.

## Start Training 

Run `sh train.sh` to train the demixing model.

(Optional) Open Tensorboard to see real-time loggings.

```bash
tensorboard --logdir .logs\ --reload_multifile True
```

# TODO

- [x] Create a new overleaf template to write the paper (here is the [link](https://www.overleaf.com/9817742265pvzhfnzxzhcq)).
- [x] Implement metrics and losses (objective functions).
- [ ] Write a run script to pre-train backbone models.
- [ ] Come up with explanation and interpretation of identity embedding (TBD).

# References

| Conference | Title | Link |
| :---: | :---: | :---: |
| | Numerical Linear Algebra | [link](http://mezbanhabibi.ir/wp-content/uploads/2020/01/NumericalLinearAlgebra-Lloyd-N1.-Trefethen-David-Bau.pdf) |
| | Calculating the Singular Values and Pseudo-Inverse of a Matrix | [link](http://www.stat.uchicago.edu/~lekheng/courses/302/classics/golub-kahan.pdf) |
| | Optimal Whitening and Decorrelation | [link](https://arxiv.org/pdf/1512.00809.pdf) |
| INTERSPEECH-20 | Joint Speaker Counting, Speech Recognition, and Speaker Identification for Overlapped Speech of Any Number of Speakers | [link](https://www.isca-speech.org/archive_v0/Interspeech_2020/pdfs/1085.pdf) |
| INTERSPEECH-21 | Stabilizing Label Assignment for Speech Separation by Self-supervised Pre-training | [link](https://www.isca-speech.org/archive/pdfs/interspeech_2021/huang21h_interspeech.pdf) |
| INTERSPEECH-21 | Streaming Multi-talker Speech Recognition with Joint Speaker Identification | [link](https://www.isca-speech.org/archive/pdfs/interspeech_2021/lu21_interspeech.pdf)

## Extra References

1. https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf
2. https://github.com/NaoyukiKanda/LibriSpeechMix
