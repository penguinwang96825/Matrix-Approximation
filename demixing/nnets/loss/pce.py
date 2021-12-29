import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


class PermutationCrossEntropy(nn.Module):

    def __init__(self, num_perm):
        super(PermutationCrossEntropy, self).__init__()
        self.num_perm = num_perm

    def forward(self, preds, targets):
        comb = permutations(range(self.num_perm), self.num_perm)
        comb = list(comb)

        # Iterate through every sample in one batch
        loss_batch = []
        for gt, pb in zip(targets, preds):
            comb = permutations(range(self.num_perm), self.num_perm)
            comb = list(comb)
            losses = []
            for c in gt[..., comb]:
                loss = torch.tensor([0.0])
                for i, c_ in enumerate(c):
                    loss += F.cross_entropy(pb[i].unsqueeze(0), c_.unsqueeze(0))
                losses.append(loss)
            loss_batch.append(min(losses))
        loss_batch = torch.stack(loss_batch)

        return loss_batch