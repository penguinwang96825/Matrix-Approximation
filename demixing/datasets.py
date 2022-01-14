import os
import random
import librosa
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from demixing.feats import MFCC
from demixing.operation import Padder2d


class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, waveforms, speakers, sr=16000, n_mfcc=20, snr=5, slice_dur=1, augmentation=0, mfcc_transform=True):
        """
        Parameters
        ----------
        waveforms: List[List[str, str]]
        speakers: List[List[int, int]]
        """
        self.sr = sr
        self.snr = snr
        self.n_mfcc = n_mfcc
        self.slice_dur = slice_dur
        self.augmentation = augmentation
        self.mfcc_transform = mfcc_transform

        if mfcc_transform:
            self.mfcc_layer = MFCC(
                sample_rate=sr, 
                n_mfcc=n_mfcc, 
                melkwargs={
                    ### Length of the FFT window ###
                    'n_fft': int(sr*0.025), 
                    ### Number of samples between successive frames ###
                    'hop_length': int(sr*0.01), 
                    ### Number of mel filterbanks ###
                    'n_mels': 80
                }
            )

        self.speakers, self.waveforms, = speakers, waveforms
        wav_mixs, self.speakers = self._generate_mixed_waveforms()
        padder = Padder2d(maxlen=sr*slice_dur)
        self.wav_mixs = padder.transform(wav_mixs)
        self.wav_mixs = torch.from_numpy(self.wav_mixs)
        self.speakers = torch.tensor(self.speakers)

    def _generate_mixed_waveforms(self):
        wav_mixs, speakers = [], []
        for (file_1, file_2), (spk_1, spk_2) in tqdm(zip(self.waveforms, self.speakers), total=len(self.waveforms)):
            wav_1, _ = librosa.load(file_1, sr=self.sr)
            wav_2, _ = librosa.load(file_2, sr=self.sr)
            wav_1_dur, wav_2_dur = len(wav_1), len(wav_2)

            wav_2_power = np.mean(np.square(wav_1)) / (10**(self.snr/10))
            scale = np.sqrt(wav_2_power / np.mean(np.square(wav_2)))

            if wav_1_dur < (self.sr * self.slice_dur):
                wav_1 = np.pad(wav_1, (0, self.sr*self.slice_dur-wav_1_dur), 'constant', constant_values=(0, 0))
                wav_1_dur = self.sr * self.slice_dur

            if wav_1_dur == wav_2_dur:
                wav_mix = wav_1 + scale * wav_2
            elif wav_1_dur > wav_2_dur:
                wav_2 = np.pad(wav_2, (0, wav_1_dur-wav_2_dur), 'constant', constant_values=(0, 0))
                wav_mix = wav_1 + scale * wav_2
            elif wav_1_dur < wav_2_dur:
                wav_mix = wav_1 + scale * wav_2[:wav_1_dur]

            # Data augmentation
            if self.augmentation > 0:
                assert isinstance(self.augmentation, int)
                if len(wav_mix) > self.sr*self.slice_dur:
                    for _ in range(self.augmentation):
                        wav_mix_ = self.random_slice(wav_mix, self.slice_dur, self.sr)
                        wav_mixs.append(wav_mix_)
                        speakers.append([spk_1, spk_2])

            wav_mixs.append(wav_mix)
            speakers.append([spk_1, spk_2])
        return wav_mixs, speakers

    @staticmethod
    def random_slice(audio, slice_dur, sr):
        slice_len = int(slice_dur * sr)
        diff = len(audio) - slice_len
        start = np.random.randint(diff)
        return audio[start:start+slice_len]

    def __len__(self):
        return len(self.wav_mixs)

    def __getitem__(self, idx):
        wav_mix = self.wav_mixs[idx, :]
        speaker_1 = self.speakers[:, 0][idx]
        speaker_2 = self.speakers[:, 1][idx]
        if self.mfcc_transform:
            # mfcc: [n_mfcc, time]
            mfcc = self.mfcc_layer(wav_mix)
            mfcc = mfcc.unsqueeze(0)
            if random.random() >= 0.5:
                mfcc = freq_mask(mfcc, F=2, num_masks=2, replace_with_zero=True)
            if random.random() >= 0.5:
                mfcc = time_mask(mfcc, T=10, num_masks=2, replace_with_zero=True)
            mfcc = mfcc.squeeze(0)
            return {
                'mfcc': mfcc, 
                'speaker_1': speaker_1, 
                'speaker_2': speaker_2
            }
        return {
            'wav_mix':wav_mix, 
            'speaker_1': speaker_1, 
            'speaker_2': speaker_2
        }


def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]
    
    for i in range(0, num_masks):        
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f) 
        if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
        else: cloned[0][f_zero:mask_end] = cloned.mean()
    
    return cloned


def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]
    
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
        else: cloned[0][:,t_zero:mask_end] = cloned.mean()
    return cloned