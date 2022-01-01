import unittest
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')


class RNNBlock(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=False):
        """
        Parameters
        ----------
        rnn_type: str
            Select from {'RNN', 'GRU', 'LSTM'}
        input_size: int
            Dimension of the input feature
        hidden_size: int
            Dimension of the hidden state
        n_layers: int
            Number of layers used in RNN
        dropout: float
            Dropout rate
        bidirectional: bool
            Whether RNN layers are bidirectional
        """
        super(RNNBlock, self).__init__()
        assert rnn_type.upper() in ['RNN', 'GRU', 'LSTM']
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = getattr(nn, rnn_type)(
            input_size, 
            hidden_size, 
            num_layers=n_layers, 
            dropout=dropout, 
            batch_first=True, 
            bidirectional=bool(bidirectional)
        )

    def forward(self, x):
        """
        x: [batch, seq_len, feat_dim]

        Returns
        -------
        RNN:
            output: [batch, seq_len, bidirectional*hidden_size]
            hn: [bidirectional*n_layers, batch, hidden_size]
        GRU: 
            output: [batch, seq_len, bidirectional*hidden_size]
            hn: [bidirectional*n_layers, batch, hidden_size]
        LSTM: 
            output: [batch, seq_len, bidirectional*hidden_size]
            hn: [bidirectional*n_layers, batch, hidden_size]
            cn: [bidirectional*n_layers, batch, hidden_size]
        """
        self.rnn.flatten_parameters()
        if self.rnn_type == "RNN":
            output, hn = self.rnn(x)
            return output, hn
        else:
            output, hn = self.rnn(x)
            return output, hn

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)
