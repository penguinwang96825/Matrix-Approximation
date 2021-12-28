import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class Padder2d(BaseEstimator, TransformerMixin):
    """
    Makes a 2d array of shape num_samples x seq length) by
    truncating or padding the list.
    
    Parameters
    ----------
    maxlen: int
        Maximum sequence length.
    pad_value: numerical (default=0)
        Value to pad with.
    dtype: dtype (default=np.float32)
        Data type of output.
    """
    def __init__(self, maxlen, pad_value=0, dtype=np.float32):
        self.maxlen = maxlen
        self.pad_value = pad_value
        self.dtype = dtype

    # pylint: disable=unused-argument
    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : list or np.ndarray
            Heterogeneous data of len n.
        
        Returns
        -------
        Xt : np.ndarray
            Homogenous array of shape (n, maxlen).
        """
        shape = len(X), self.maxlen
        Xt = self.pad_value * np.ones(shape, dtype=self.dtype)

        for i, arr in enumerate(X):
            m = min(self.maxlen, len(arr))
            if not m:
                continue
            arr = np.array(arr[:m])
            Xt[i, :m] = arr
        return Xt