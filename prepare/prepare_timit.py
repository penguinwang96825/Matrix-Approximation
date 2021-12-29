import os
import re
import sys
import json
import shutil
import zipfile
import argparse
import librosa
import requests
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from urllib.request import urlopen
from sklearn.model_selection import train_test_split


URL = 'https://data.deepai.org/timit.zip'
PROJECT_ROOT = Path(os.path.abspath(os.getcwd()))
TIMIT_CORPUS_ROOT = os.path.join(PROJECT_ROOT, 'data', 'corpus', 'timit')
TEST_PATH = os.path.join(TIMIT_CORPUS_ROOT, 'TEST')
TRAIN_PATH = os.path.join(TIMIT_CORPUS_ROOT, 'TRAIN')
TIMIT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "dataset", "timit")


logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class TIMIT(object):
    """
    Step 1. Download TIMIT from https://data.deepai.org/timit.zip
    Step 2. Extract TRAIN and TEST folders to ./data/corpus/timit/
    """
    def __init__(self, timit_root_directory, store_path, skip_sa=True):
        super(TIMIT).__init__()
        self.timit_root_folder = timit_root_directory
        self.store_path = store_path
        self.skip_sa = skip_sa

    def build(self):
        spk_files_train, spk_id_train = self.build_speaker_mapping(mode='TRAIN')
        spk_files_test, spk_id_test = self.build_speaker_mapping(mode='TEST')

        with open(f'{self.store_path}/timit.speaker.train.txt', 'w') as f:
            for file_path, spk in zip(spk_files_train, spk_id_train):
                to_write = f'{file_path}\t{spk}'
                f.write(f'{to_write}\n')
        with open(f'{self.store_path}/timit.speaker.test.txt', 'w') as f:
            for file_path, spk in zip(spk_files_test, spk_id_test):
                to_write = f'{file_path}\t{spk}'
                f.write(f'{to_write}\n')

        speakers = sorted(set(spk_id_train+spk_id_test))
        with open(f'{self.store_path}/speaker.map.txt', 'w') as f:
            for idx, speaker in enumerate(speakers):
                to_write = f'{speaker}\t{idx}'
                f.write(f'{to_write}\n')

    def build_speaker_mapping(self, mode='TRAIN'):
        root_folder = os.path.join(self.timit_root_folder, mode)
        timit_data = []
        for dialect_region in os.listdir(root_folder):
            dialect_region_dir_path = os.path.join(root_folder, dialect_region)
            for speaker_id in os.listdir(dialect_region_dir_path):
                speaker_id_dir_path = os.path.join(dialect_region_dir_path, speaker_id)
                for file in os.listdir(speaker_id_dir_path):
                    if file.endswith("WAV"):
                        id_ = file.split(".")[0]
                        sentence_type = re.findall("[A-Za-z]+", id_.strip())[0]
                        file_path = os.path.join(speaker_id_dir_path, file)
                        timit_data.append([
                            dialect_region, 
                            file_path, 
                            id_, 
                            sentence_type, 
                            speaker_id
                        ])
        timit_data = pd.DataFrame(
            timit_data, 
            columns=["dialect_region", "file", "id", "sentence_type", "speaker_id"]
        )
        timit_data = timit_data.sort_values("speaker_id").reset_index(drop=True)
        if self.skip_sa:
            timit_data = timit_data.query("sentence_type!='SA'")
        return timit_data['file'].tolist(), timit_data['speaker_id'].tolist()


class TIMIT2Mix(object):
    """
    Each speaker has 10 utterances
        - 2 SA (dialect sentences)
        - 5 SX (phonetically compact sentences)
        - 3 SI (phonetically diverse sentences): the Brown Corpus + the Playwrights Dialog

    According to TIMIT official documentation, they recommend splitting 70% training and 30% testing.
    The format of the audio path name is below:
        /<corpus>/<split>/<dialect>/<speakerID>/<utteranceID>.<extension>

    References
    ----------
    1. https://perso.limsi.fr/lamel/TIMIT_NISTIR4930.pdf
    2. https://data.deepai.org/timit.zip
    """
    def __init__(self, store_path):
        self.store_path = store_path
        self.load_mapping()

        paths_train, speakers_train, durs_train = self.read(mode='train')
        paths_test, speakers_test, durs_test = self.read(mode='test')
        paths_train, paths_test, speakers_train, speakers_test, durs_train, durs_test = self.resplit(
            paths_train, paths_test, speakers_train, speakers_test, durs_train, durs_test
        )

        self.paths_train = paths_train
        self.paths_test = paths_test
        self.speakers_train = speakers_train
        self.speakers_test = speakers_test
        self.durs_train = durs_train
        self.durs_test = durs_test

    def load_mapping(self):
        self.speaker2idx, self.idx2speaker = {}, {}
        with open(f'{self.store_path}/speaker.map.txt', 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                speaker, idx = line.split()
                speaker, idx = str(speaker), int(idx)
                self.speaker2idx[speaker] = idx
                self.idx2speaker[idx] = speaker

    def resplit(self, X_train, X_test, y_train, y_test, z_train, z_test):
        X = X_train + X_test
        y = y_train + y_test
        z = z_train + z_test
        X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
            X, y, z, test_size=0.3, random_state=914, stratify=y
        )
        return X_train, X_test, y_train, y_test, z_train, z_test

    def mix(self, num_mix):
        train_generator = self.joint_different_speakers(self.paths_train, self.speakers_train, self.durs_train, num_mix)
        with open(os.path.join(self.store_path, f'train-clean-{num_mix}mix.jsonl'), 'w') as f:
            for trg_file, itf_file, trg_spk, itf_spk, trg_dur, itf_dur in train_generator:
                observation = {}
                observation['speakers'] = [trg_spk, itf_spk]
                observation['waveforms'] = [trg_file, itf_file]
                observation['durations'] = [trg_dur, itf_dur]
                f.write(json.dumps(observation) + "\n")

        test_generator = self.joint_different_speakers(self.paths_test, self.speakers_test, self.durs_test, num_mix)
        with open(os.path.join(self.store_path, f'test-clean-{num_mix}mix.jsonl'), 'w') as f:
            for trg_file, itf_file, trg_spk, itf_spk, trg_dur, itf_dur in test_generator:
                observation = {}
                observation['speakers'] = [trg_spk, itf_spk]
                observation['waveforms'] = [trg_file, itf_file]
                observation['durations'] = [trg_dur, itf_dur]
                f.write(json.dumps(observation) + "\n")

    def joint_different_speakers(self, audio_files:List[str], speakers:List[str], durations:List[float], num_mix:int):
        """
        Parameters
        ----------
        audio_files: List[str]
        speakers: List[str]
        durations: List[float]
        num_mix: int
        """
        for i, file_ in enumerate(audio_files):
            current_speaker = speakers[i]
            current_dur = durations[i]
            is_different_speakers = list(map(lambda x: x!=current_speaker, speakers))
            different_speakers_idx = [k for k, boolean in enumerate(is_different_speakers) if boolean]
            select_idx = list(np.random.choice(different_speakers_idx, num_mix, replace=False))
            for j in select_idx:
                trg_file = file_
                itf_file = audio_files[j]
                trg_spk = current_speaker
                itf_spk = speakers[j]
                trg_dur = current_dur
                itf_dur = durations[j]
                yield trg_file, itf_file, trg_spk, itf_spk, trg_dur, itf_dur

    def read(self, mode='train'):
        paths, speakers, durs = [], [], []
        with open(f'{self.store_path}/timit.speaker.{mode}.txt', 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                path, speaker = line.split()
                dur = librosa.get_duration(filename=path)
                paths.append(path)
                speakers.append(self.speaker2idx[speaker])
                durs.append(dur)
        return paths, speakers, durs


def prepare_timit_corpus():

    def download_():
        if not (check_folders(TRAIN_PATH) and check_folders(TEST_PATH)):
            logger.info('TIMIT corpus not exist, start downloading...')
            try:
                os.makedirs(TIMIT_CORPUS_ROOT)
            except:
                pass
            dst_file_path = os.path.join(TIMIT_CORPUS_ROOT, 'timit.zip')
            download_url(URL, dst_file_path)
            dst_file_path = os.path.join(TIMIT_CORPUS_ROOT, 'timit.zip')
            with zipfile.ZipFile(dst_file_path, 'r') as zip_ref:
                zip_ref.extractall(TIMIT_CORPUS_ROOT)
            TEMP_PATH = os.path.join(TIMIT_CORPUS_ROOT, 'data')
            shutil.move(
                os.path.join(TEMP_PATH, 'TRAIN'),
                TIMIT_CORPUS_ROOT
            )
            shutil.move(
                os.path.join(TEMP_PATH, 'TEST'),
                TIMIT_CORPUS_ROOT
            )
            os.remove(dst_file_path)
            try:
                shutil.rmtree(TEMP_PATH)
            except OSError as e:
                logger.info("Error: %s : %s" % (TEMP_PATH, e.strerror))
        else:
            logger.info('TIMIT corpus already exist!')

    download_()


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def _progress_bar(count, total):
    """Report download progress.
    Credit:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(
        '  [{}] {}% of {:.1f}MB file  \r'.
        format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write('\n')


def download_url(url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar):
    """Download url and write it to dst_file_path.
    Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    response = urlopen(url)
    total_size = requests.head(url).headers['content-length'].strip()
    total_size = int(total_size)
    bytes_so_far = 0

    with open(dst_file_path, 'wb') as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)

    return bytes_so_far


def main():
    args = argparse.ArgumentParser(
        description="Preparing TIMIT dataset for training."
    )
    args.add_argument(
        "-m",
        "--num_mix",
        default=2,
        type=int,
        help="Number of mixture",
    )
    args = args.parse_args()

    prepare_timit_corpus()

    # Build TIMIT dataset (for speaker ID task purpose)
    logger.info('Build TIMIT dataset (for speaker ID task purpose)')
    timit = TIMIT(TIMIT_CORPUS_ROOT, TIMIT_DATASET_ROOT, skip_sa=True)
    timit.build()

    # Build TIMIT mixture dataset (for speaker demixing task purpose)
    logger.info('Build TIMIT mixture dataset (for speaker demixing task purpose)')
    timit_2mix = TIMIT2Mix(TIMIT_DATASET_ROOT)
    timit_2mix.mix(num_mix=args.num_mix)


if __name__ == '__main__':
    main()