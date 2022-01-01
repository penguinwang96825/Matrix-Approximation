import os
import sys
import shutil
import requests
from pathlib import Path
from urllib.request import urlopen


MINI_LIBRISPEECH_TRAIN_URL = "http://www.openslr.org/resources/31/train-clean-5.tar.gz"
PROJECT_ROOT = Path(os.path.abspath(os.getcwd()))
MINI_LIBRISPEECH_CORPUS_ROOT = os.path.join(PROJECT_ROOT, 'data', 'corpus', 'mini_librispeech')


def download_mini_librispeech():

    def download_(destination):
        train_archive = os.path.join(destination, "train-clean-5.tar.gz")
        download_url(MINI_LIBRISPEECH_TRAIN_URL, train_archive)
        shutil.unpack_archive(train_archive, destination)

    if check_folders(MINI_LIBRISPEECH_CORPUS_ROOT):
        download_(MINI_LIBRISPEECH_CORPUS_ROOT)
        TRAIN_ARCHIVE = os.path.join(MINI_LIBRISPEECH_CORPUS_ROOT, "train-clean-5.tar.gz")
        TEMP_PATH = os.path.join(MINI_LIBRISPEECH_CORPUS_ROOT, 'LibriSpeech')
        shutil.move(
            os.path.join(TEMP_PATH, 'train-clean-5'),
            MINI_LIBRISPEECH_CORPUS_ROOT
        )
        SOURCE_DIR = TEMP_PATH
        TARGET_DIR = MINI_LIBRISPEECH_CORPUS_ROOT
        file_names = os.listdir(SOURCE_DIR)
        for file_name in file_names:
            shutil.move(os.path.join(SOURCE_DIR, file_name), TARGET_DIR)
        os.system(f'rm -r -f {SOURCE_DIR}')
        os.system(f'rm -r -f {TRAIN_ARCHIVE}')


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


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def main():
    download_mini_librispeech()


if __name__ == '__main__':
    main()