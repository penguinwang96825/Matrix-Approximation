# Matrix Factorisation

We propose a novel de-mixing scheme which works by singular value decomposition (SVD) decorrelating in embedding domain. With the help of SVD properties, this approach is better than performing de-mixing functions to extract the target speakers' embeddings. Compared to de-mixing functions, the presented approach doesn't require pre-computed speaker inventory containing profile vector (d-vector) of the speaker. Instead of utilising SVD for the purposes of signal reconstruction or data compression, the proposed approach explains statistical variation among attributes for building an orthonormal embedding space. This interpretation is what makes it useful in the context of signal processing.

# Matrix Decomposition

## SVD

SVD and PCA aim to find linearly uncorrelated orthonormal axes. Since we systematically exhaust the structure, at the end we are left mostly with noise. The common wisdom is that we just keep the first few linear combination (1 to 4) and drop the rest.

* Top-k components are due to structure.
* The other components are due to noise.

Compute a faster SVD:

1. Golub-Kahan Bidiagonalization algorithm
2. Lawson-Hanson-Chan algorithm

## NoS (Number of Speakers) Estimation

1. Window-Disjoint Orthogonal (WDO)
2. Density-based Clustering

# Start Training 

Run the command line below.

```bash

```

(Optional) Open Tensorboard to see real-time loggings.

```bash
tensorboard --logdir .logs\ --reload_multifile True
```

## Configuration

```
#############################################################
# Consider the audio signal to be a time series sampled at  #
# an interval of 25ms with step size of 10ms                #
#                                                           #
# 25ms    25ms   25ms   25ms …  Frames                      #
# 400     400    400    400  …  Samples/Frame               #
#                                                           #
# |—————|—————|—————|—————|—————|—————|—————|————-|         #
#                                                           #
# |—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—| #
#                                                           #
# 10 10 10 10 … Frame Step                                  #
#############################################################
```

# Others

Up-to-date repository can be found from [here](https://github.com/asifjalal/speaker-embedding-factorisation).

# References

1. http://mezbanhabibi.ir/wp-content/uploads/2020/01/NumericalLinearAlgebra-Lloyd-N1.-Trefethen-David-Bau.pdf
2. http://www.stat.uchicago.edu/~lekheng/courses/302/classics/golub-kahan.pdf
3. https://arxiv.org/pdf/1512.00809.pdf
4. https://arxiv.org/pdf/2006.10930.pdf
5. https://arxiv.org/pdf/2010.15366v3.pdf
6. https://arxiv.org/pdf/2104.02109.pdf
7. https://github.com/NaoyukiKanda/LibriSpeechMix
8. https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf