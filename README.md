# Matrix Factorisation

We propose a novel de-mixing scheme which works by singular value decomposition (SVD) decorrelating in embedding domain. With the help of SVD properties, this approach is better than performing de-mixing functions to extract the target speakers' embeddings. Compared to de-mixing functions, the presented approach doesn't require pre-computed speaker inventory $\mathcal{D} = {d_1, \ldots, d_k}$, where $k$ is the number of speakers in the inventory and $d_k$ is a speaker profile vector of the $k$-th speaker. Instead of utilising SVD for signal reconstruction or data compression, the proposed approach explains statistical variation among attributes for building an orthonormal embedding space. This interpretation is what makes it useful in the context of signal processing.

# Matrix Decomposition

## SVD

SVD and PCA aim to find linearly uncorrelated orthonormal axes.

Compute a faster SVD:

1. Golub-Kahan Bidiagonalization algorithm
2. Lawson-Hanson-Chan algorithm

## NoS (Number of Speakers) Estimation

1. Window-Disjoint Orthogonal (WDO)
2. Density-based Clustering

# References

1. http://mezbanhabibi.ir/wp-content/uploads/2020/01/NumericalLinearAlgebra-Lloyd-N1.-Trefethen-David-Bau.pdf
2. http://www.stat.uchicago.edu/~lekheng/courses/302/classics/golub-kahan.pdf
3. https://arxiv.org/pdf/1512.00809.pdf