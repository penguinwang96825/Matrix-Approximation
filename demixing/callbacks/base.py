import os
import torch
import fsspec
from pathlib import Path
from functools import wraps
from typing import Union, Optional, Callable, Any
from fsspec.implementations.local import AbstractFileSystem, LocalFileSystem
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from .callback import Callback
from ._rank import rank_zero_experiment, rank_zero_only


class BaseLogger(object):

    def __init__(
        self, 
        log_dir=".logs/", 
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[str] = None, 
        **kwargs
    ):
        self._save_dir = log_dir
        self._name = name or ""
        self._version = version
        self._sub_dir = sub_dir
        self._log_graph = log_graph
        self._default_hp_metric = default_hp_metric
        self._prefix = prefix
        self._fs = self.get_filesystem(log_dir)

        self._experiment = None
        self.hparams = {}
        self._kwargs = kwargs

    @property
    def root_dir(self) -> str:
        if self.name is None or len(self.name) == 0:
            return self.save_dir
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def sub_dir(self) -> Optional[str]:
        return self._sub_dir

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = self.root_dir

        try:
            listdir_info = self._fs.listdir(root_dir)
        except OSError:
            print("Missing logger folder: %s", root_dir)
            return 0

        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if self._fs.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @property
    @rank_zero_experiment
    def experiment(self) -> SummaryWriter:
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_tensorboard_function()

        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._experiment

    def get_filesystem(self, path: Union[str, Path]) -> AbstractFileSystem:
        path = str(path)
        if "://" in path:
            # use the fileystem from the protocol specified
            return fsspec.filesystem(path.split(":", 1)[0])
        # use local filesystem
        return LocalFileSystem()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.experiment.flush()
        self.experiment.close()
        self.save()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state