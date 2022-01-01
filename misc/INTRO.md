# Matrix Decomposition

## SVD

SVD and PCA aim to find linearly uncorrelated orthonormal axes. Since we systematically exhaust the structure, at the end we are left mostly with noise. The common wisdom is that we just keep the first few linear combination (1 to 4) and drop the rest.

* Top-k components are due to structure.
* The other components are due to noise.

Compute a faster SVD:

1. Golub-Kahan Bidiagonalization algorithm
2. Lawson-Hanson-Chan algorithm

A detailed SVD explanation can be found [here](https://www2.math.ethz.ch/education/bachelor/lectures/hs2014/other/linalg_INFK/svdneu.pdf) and [here](https://mml-book.github.io/book/mml-book.pdf) on page 134.

## NoS (Number of Speakers) Estimation

1. Window-Disjoint Orthogonal (WDO)
2. Density-based Clustering

# Feature Setup

We start with a speech signal, and we'll assume sampled at 16kHz. Frame the signal into 20-40 ms frames (25ms is standard). Frame step is usually something like 10ms (160 samples), which allows some overlap to the frames.

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