import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from sklearn import metrics
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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
set_seed(CONFIG['seed'])


class DecompositionNet(Module):

    def __init__(self, n_mfcc, emb_dim, dropout, max_speakers, num_classes):
        super(DecompositionNet, self).__init__()
        self.encoder = XVector(n_mfcc, dropout)
        self.expand = ResidualMLP(1, max_speakers, layers=[64, 32, 16, 8])
        self.ff_u1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(emb_dim, emb_dim)), 
            ('bn1', nn.BatchNorm1d(emb_dim)), 
            ('dropout1', nn.Dropout(p=dropout)), 
            ('tanh', nn.Tanh()), 
            ('fc2', nn.Linear(emb_dim, emb_dim)), 
            ('bn2', nn.BatchNorm1d(emb_dim)), 
            ('dropout2', nn.Dropout(p=dropout)), 
            ('sigmoid', nn.Sigmoid()), 
            ('fc3', nn.Linear(emb_dim, emb_dim))
        ]))
        self.ff_u2 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(emb_dim, emb_dim)), 
            ('bn1', nn.BatchNorm1d(emb_dim)), 
            ('dropout1', nn.Dropout(p=dropout)), 
            ('tanh', nn.Tanh()), 
            ('fc2', nn.Linear(emb_dim, emb_dim)), 
            ('bn2', nn.BatchNorm1d(emb_dim)), 
            ('dropout2', nn.Dropout(p=dropout)), 
            ('sigmoid', nn.Sigmoid()), 
            ('fc3', nn.Linear(emb_dim, emb_dim))
        ]))
        self.decomposer = SVD()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(emb_dim*2, emb_dim)), 
            ('bn1', nn.BatchNorm1d(emb_dim)), 
            ('dropout1', nn.Dropout(p=dropout)), 
            ('relu', nn.LeakyReLU(inplace=True)), 
            ('fc2', nn.Linear(emb_dim, num_classes))
        ]))

        self.step_scheduler_after = "batch"

    def forward(self, mfcc, speaker_1=None, speaker_2=None):
        # mfcc: [batch, n_mfcc, time]
        mfcc = mfcc.transpose(1, 2)
        # mixed_emb: [batch, emb_dim]
        mixed_emb = self.encoder(mfcc)
        # mixed_emb:[batch, emb_dim, 1]
        mixed_emb = mixed_emb[:, :, None]
        # M: [batch, emb_dim, max_speakers]
        M = self.expand(mixed_emb)
        # U: [batch, emb_dim, max_speakers]
        # S: [batch, max_speakers, max_speakers]
        # V: [batch, max_speakers, max_speakers]
        U, S, V = self.decomposer(M)
        # pca: [batch, emb_dim*2]
        pca = torch.matmul(U, S)[:, :, :2]
        pca = nn.Flatten()(pca)
        # outputs: [batch, num_classes]
        outputs = self.classifier(pca)
        outputs = torch.sigmoid(outputs)

        # targets: [batch, num_classes]
        targets = F.one_hot(speaker_1, num_classes=630) + F.one_hot(speaker_2, num_classes=630)
        loss = self.loss_fn(outputs, targets.float())
        m = self.monitor_metrics(outputs, targets)
        return None, loss, m

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        return {fn.__name__:fn(targets, outputs>=0.5) for fn in self.metrics_fn}


def hamming_score(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def main():
    NUM_TO_DEMIX = 2

    TRAIN_JSONL_PATH = os.path.join(TIMIT_DATASET_ROOT, 'train-clean-2mix.jsonl')
    ds = SpeechDataset(
        TRAIN_JSONL_PATH, 
        sr=CONFIG['sample_rate'], 
        n_mfcc=CONFIG['n_mfcc'], 
        snr=CONFIG['snr'], 
        slice_dur=CONFIG['slice_dur'], 
        augmentation=CONFIG['augmentation'], 
        mfcc_transform=True
    )
    train_ds, valid_ds = torch.utils.data.random_split(ds, [round(len(ds)*.8), round(len(ds)*.2)])

    print(
        f'TRAIN: {len(train_ds)}\t'
        f'VALID: {len(valid_ds)}'
    )

    tb = TensorBoardLogger(log_dir=".logs", name='svd')
    model = DecompositionNet(
        n_mfcc=CONFIG['n_mfcc'], 
        emb_dim=CONFIG['emb_size'], 
        dropout=CONFIG['dropout'], 
        max_speakers=5, 
        num_classes=630
    )
    loss_fn = nn.BCELoss()
    total_step = len(train_ds) / CONFIG['train_batch_size'] * CONFIG['number_of_epochs']
    optimiser = CONFIG['opt_class'](model.parameters())
    scheduler = CONFIG['sch_class'](optimiser)
    # scheduler = None
    
    model.compile(
        loss_fn=loss_fn, 
        optimizer=optimiser, 
        metrics_fn=[metrics.accuracy_score, hamming_score], 
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