import os
import argparse
import torch
import json
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
CHECKPOINTS_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")
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

    def forward(self, batch):
        # mfcc: [batch, n_mfcc, time]
        mfcc = batch['mfcc'].transpose(1, 2)
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

        return outputs

    def compute_objectives(self, output, batch):
        # targets: [batch, num_classes]
        speaker_1, speaker_2 = batch['speaker_1'], batch['speaker_2']
        targets = F.one_hot(speaker_1, num_classes=630) + F.one_hot(speaker_2, num_classes=630)
        return self.loss_fn(output, targets.float())

    def compute_metrics(self, output, batch):
        # targets: [batch, num_classes]
        speaker_1, speaker_2 = batch['speaker_1'], batch['speaker_2']
        targets = F.one_hot(speaker_1, num_classes=630) + F.one_hot(speaker_2, num_classes=630)
        return self.monitor_metrics(output, targets)

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


def load_jsonl(jsonl_path):
    speakers, waveforms = [], []
    with open(jsonl_path, 'r') as f:
        for line in f:
            observation = json.loads(line)
            speakers.append(observation['speakers'])
            waveforms.append(observation['waveforms'])
    return waveforms, speakers


def main():
    args = argparse.ArgumentParser(
        description="Preparing TIMIT dataset for training."
    )
    args.add_argument(
        "-a",
        "--augmentation",
        default=0,
        type=int,
        help="Number of augmentation",
    )
    args = args.parse_args()

    # NUM_TO_DEMIX = 2
    CHECKPOINTS_PATH = os.path.join(CHECKPOINTS_ROOT, f'svd-aug{args.augmentation}-bs128-lr3-adamw-noam.ckpt')
    TRAIN_JSONL_PATH = os.path.join(TIMIT_DATASET_ROOT, 'train-clean-2mix.jsonl')
    TEST_JSONL_PATH = os.path.join(TIMIT_DATASET_ROOT, 'test-clean-2mix.jsonl')

    X_train, y_train = load_jsonl(TRAIN_JSONL_PATH)
    X_test, y_test = load_jsonl(TEST_JSONL_PATH)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=914)

    train_ds = SpeechDataset(
        X_train, y_train, 
        sr=CONFIG['sample_rate'], 
        n_mfcc=CONFIG['n_mfcc'], 
        snr=CONFIG['snr'], 
        slice_dur=CONFIG['slice_dur'], 
        augmentation=args.augmentation, 
        mfcc_transform=True
    )
    valid_ds = SpeechDataset(
        X_valid, y_valid, 
        sr=CONFIG['sample_rate'], 
        n_mfcc=CONFIG['n_mfcc'], 
        snr=CONFIG['snr'], 
        slice_dur=CONFIG['slice_dur'], 
        augmentation=0, 
        mfcc_transform=True
    )
    test_ds = SpeechDataset(
        X_test, y_test, 
        sr=CONFIG['sample_rate'], 
        n_mfcc=CONFIG['n_mfcc'], 
        snr=CONFIG['snr'], 
        slice_dur=CONFIG['slice_dur'], 
        augmentation=5, 
        mfcc_transform=True
    )

    print(
        f'TRAIN: {len(train_ds)}\t'
        f'VALID: {len(valid_ds)}\t'
        f'TEST: {len(test_ds)}'
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
    model.save(os.path.join(CHECKPOINTS_ROOT, f'svd-aug{args.augmentation}-bs128-lr3-adamw-noam.ckpt'), weights_only=False)

    prediction = model.predict(test_ds, batch_size=256, n_jobs=0)
    prediction = np.vstack(list(prediction))
    
    targets = F.one_hot(test_ds[:]['speaker_1'], num_classes=630) + F.one_hot(test_ds[:]['speaker_2'], num_classes=630)
    acc = metrics.accuracy_score(targets.cpu().detach().numpy(), prediction>=0.5)
    ham = hamming_score(targets.cpu().detach().numpy(), prediction>=0.5)
    print(
        f'Accuracy: {acc}'
        f'Hamming: {ham}'
    )


if __name__ == '__main__':
    main()