import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


class PermutationCrossEntropy(nn.Module):

    def __init__(self, num_perm, device='cuda'):
        super(PermutationCrossEntropy, self).__init__()
        self.num_perm = num_perm
        self.device = device

    def get_combinations(self):
        comb = permutations(range(self.num_perm), self.num_perm)
        comb = list(comb)
        return comb

    def forward(self, preds, targets):
        """
        Parameters
        ----------
        preds: torch.tensor
            Size of [batch_size, num_speaker_to_demix, num_classes]
        targets: torch.tensor
            Size of [batch_size, num_speaker_to_demix]

        Examples
        --------
        >>> BATCH_SIZE = 2
        >>> NUM_CLASSES = 5
        >>> NUM_SPEAKER_TO_DEMIX = 3
        >>> y_true = torch.LongTensor([
        ...     [0, 1, 2], 
        ...     [0, 2, 4]
        ... ])
        >>> y_pred = torch.FloatTensor([
        ...     [
        ...         [0.1, 0.2, 0.3, 0.4, 0.5], 
        ...         [0.4, 0.5, 0.1, 0.6, 0.1], 
        ...         [0.6, 0.4, 0.3, 0.3, 0.5]
        ...     ], 
        ...     [
        ...         [0.7, 0.6, 0.5, 0.4, 0.3], 
        ...         [0.1, 0.3, 0.5, 0.7, 0.9], 
        ...         [0.5, 0.7, 0.9, 0.1, 0.3]
        ...     ]
        ... ])
        >>> pce = PermutationCrossEntropy(NUM_SPEAKER_TO_DEMIX)
        >>> loss = pce(y_pred, y_true)
        ... tensor([[4.5261], 
        ...         [3.9176]])
        """
        # Iterate through every sample in one batch
        comb = self.get_combinations()
        loss_batch = []
        targets, preds = targets.to(self.device), preds.to(self.device)
        for gt, pb in zip(targets, preds):
            losses = []
            for c in gt[..., comb]:
                loss = torch.tensor([0.0], device=self.device)
                for i, c_ in enumerate(c):
                    loss += F.cross_entropy(pb[i].unsqueeze(0), c_.unsqueeze(0))
                losses.append(loss)
            loss_batch.append(min(losses))
        loss_batch = torch.stack(loss_batch)
        return torch.mean(loss_batch)