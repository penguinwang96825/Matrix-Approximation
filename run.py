import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import OrderedDict
from demixing import set_seed
from demixing.datasets import SpeechDataset
from demixing.solver import Module
from demixing.nnets import XVector
from demixing.nnets import SVD
from demixing.nnets import ResidualMLP
from demixing.nnets.loss import PermutationCrossEntropy
from demixing.callbacks import TensorBoardLogger
from demixing import load_hyperpyyaml


with open("./config.yaml") as f:
    CONFIG = load_hyperpyyaml(f)


PROJECT_ROOT = Path(os.path.abspath(os.getcwd()))
TIMIT_CORPUS_ROOT = os.path.join(PROJECT_ROOT, "data", "corpus", "timit")
TIMIT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "dataset", "timit")
set_seed(CONFIG['seed'])


def permutation_accuracy_pytorch(y_true, y_pred):

    def intersect(x, y):
        x_cat_y, counts = torch.cat([x, y]).unique(return_counts=True)
        return x_cat_y[torch.where(counts.gt(1))]

    total, correct = 0, 0
    for t, p in zip(y_true, y_pred):
        correct += len(intersect(t, p))
        total += 1
    return correct / (2*total)


def permutation_accuracy_numpy(y_true, y_pred):
    total, correct = 0, 0
    for t, p in zip(y_true, y_pred):
        correct += len(np.intersect1d(t, p))
        total += 1
    return correct / (2*total)


class DecompositionNet(Module):

    def __init__(self, n_mfcc, emb_dim, dropout, max_speakers, num_classes):
        super(DecompositionNet, self).__init__()
        self.encoder = XVector(n_mfcc, dropout)
        self.expand = ResidualMLP(1, max_speakers, layers=[25, 15])
        self.ff_u1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(emb_dim, emb_dim)), 
            ('bn1', nn.BatchNorm1d(emb_dim)), 
            ('dropout1', nn.Dropout(p=dropout)), 
            ('relu1', nn.Tanh()), 
            ('fc2', nn.Linear(emb_dim, emb_dim)), 
            ('bn2', nn.BatchNorm1d(emb_dim)), 
            ('dropout2', nn.Dropout(p=dropout)), 
            ('relu2', nn.Sigmoid())
        ]))
        self.ff_u2 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(emb_dim, emb_dim)), 
            ('bn1', nn.BatchNorm1d(emb_dim)), 
            ('dropout1', nn.Dropout(p=dropout)), 
            ('relu1', nn.Tanh()), 
            ('fc2', nn.Linear(emb_dim, emb_dim)), 
            ('bn2', nn.BatchNorm1d(emb_dim)), 
            ('dropout2', nn.Dropout(p=dropout)), 
            ('relu2', nn.Sigmoid())
        ]))
        self.decomposer = SVD()
        self.classifier = nn.Linear(emb_dim, num_classes)

        self.step_scheduler_after = "batch"

    def forward(self, mfcc, speaker_1=None, speaker_2=None):
        # mfcc: [batch, n_mfcc, time]
        mfcc = mfcc.transpose(1, 2)
        # mixed_emb: [batch, emb_dim=512]
        mixed_emb = self.encoder(mfcc)
        # mixed_emb:[batch, emb_dim=512, 1]
        mixed_emb = mixed_emb[:, :, None]
        # M: [batch, emb_dim, max_speakers]
        M = self.expand(mixed_emb)
        # U: [batch, emb_dim, max_speakers]
        # S: [batch, max_speakers, max_speakers]
        # V: [batch, max_speakers, max_speakers]
        U, S, V = self.decomposer(M)
        # u_1: [batch, emb_dim]
        # u_2: [batch, emb_dim]
        u_1, u_2 = U[:, :, 0], U[:, :, 1]
        # u_1: [batch, emb_dim]
        u_1 = self.ff_u1(u_1) + u_1
        # u_2: [batch, emb_dim]
        u_2 = self.ff_u2(u_2) + u_2
        # output_1: [batch, num_classes]
        pred_1 = self.classifier(u_1)
        class_1 = torch.argmax(pred_1, dim=1)
        # output_2: [batch, num_classes]
        pred_2 = self.classifier(u_2)
        class_2 = torch.argmax(pred_2, dim=1)
        # output: [batch, num_to_demix, num_classes]
        preds = torch.stack((pred_1, pred_2), dim=1)
        # tagets: [batch, num_to_demix]
        targets = torch.stack((speaker_1, speaker_2))
        predictions = torch.stack((class_1, class_2), dim=1)
        loss = self.loss_fn(preds, targets)
        m = {fn.__name__:fn(predictions.to('cpu'), targets.to('cpu')) for fn in self.metrics_fn}
        return None, loss, m


def main():
    TRAIN_JSONL_PATH = os.path.join(TIMIT_DATASET_ROOT, 'train-clean-2mix.jsonl')
    ds = SpeechDataset(TRAIN_JSONL_PATH, mfcc_transform=True)
    train_ds, valid_ds = torch.utils.data.random_split(ds, [round(len(ds)*.8), round(len(ds)*.2)])
    train_ds = train_ds.dataset
    valid_ds = valid_ds.dataset

    tb = TensorBoardLogger(log_dir=".logs", name='svd')
    model = DecompositionNet(
        n_mfcc=CONFIG['n_mfcc'], 
        emb_dim=CONFIG['emb_size'], 
        dropout=CONFIG['dropout'], 
        max_speakers=10, 
        num_classes=630
    )
    loss_fn = PermutationCrossEntropy(num_perm=2).to(CONFIG['device'])
    total_step = len(train_ds) / CONFIG['train_batch_size'] * CONFIG['number_of_epochs']
    optimiser = CONFIG['opt_class'](model.parameters())
    scheduler = CONFIG['sch_class'](optimiser)
    
    model.compile(
        loss_fn=loss_fn, 
        optimizer=optimiser, 
        metrics_fn=[permutation_accuracy_numpy], 
        scheduler=scheduler
    )
    model.fit(
        train_dataset=train_ds, 
        valid_dataset=valid_ds, 
        train_bs=CONFIG['train_batch_size'], 
        valid_bs=CONFIG['valid_batch_size'], 
        device=CONFIG['device'], 
        epochs=CONFIG['number_of_epochs'], 
        n_jobs=CONFIG['n_jobs'], 
        fp16=False, 
        callbacks=[tb]
    )


if __name__ == '__main__':
    main()