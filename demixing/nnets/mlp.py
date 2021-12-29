import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Parameters
    ----------
    input_dim: int
    output_dim: int
    layers: List[int]
    """
    def __init__(self, input_dim, output_dim, layers):
        super(MLP, self).__init__()

        model = nn.Sequential()
        model.add_module(
            'initial-lin',
            nn.Linear(input_dim, layers[0])
        )
        model.add_module(
            'initial-relu',
            nn.ReLU()
        )

        for i in range(len(layers)-1):
            model.add_module(
                'layer-{}lin'.format(i+1),
                nn.Linear(layers[i], layers[i+1])
            )
            model.add_module(
                'layer-{}relu'.format(i+1),
                nn.ReLU()
            )

        model.add_module(
            'final-lin',
            nn.Linear(layers[-1], output_dim)
        )

        self.model = model

    def forward(self, x):
        return self.model(x)


class ResidualMLP(nn.Module):
    """
    MLP with residual downsampled connection

    Parameters
    ----------
    input_dim: int
    output_dim: int
    layers: List[int]
    """
    def __init__(self, input_dim:int, output_dim:int, layers:List[int]):
        super(ResidualMLP, self).__init__()

        self.downsample = nn.Linear(input_dim, output_dim)

        model = nn.Sequential()
        model.add_module(
            'initial-layer-lin',
            nn.Linear(input_dim, layers[0])
        )
        model.add_module(
            'initial-layer-relu',
            nn.ReLU()
        )

        for i in range(len(layers)-1):
            model.add_module(
                'layer-{}-lin'.format(i+1),
                nn.Linear(layers[i], layers[i+1])
            )
            model.add_module(
                'layer-{}-relu'.format(i+1),
                nn.ReLU()
            )

        model.add_module(
            'last-layer-lin',
            nn.Linear(layers[-1], output_dim)
        )

        self.model = model


    def forward(self, x):
        return self.downsample(x) + self.model(x)