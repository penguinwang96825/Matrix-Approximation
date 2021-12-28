import os
import sys
import torch
import random
import numpy as np
from urllib.request import urlopen


def set_seed(seed=914, cuda=torch.cuda.is_available()):
    """ Set seed values for random generators.
    Parameters
    ----------
    seed: int 
        Value used as a seed.
    cuda bool (optional)
        Whether to set the cuda seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
    total_size = response.info().getheader('Content-Length').strip()
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