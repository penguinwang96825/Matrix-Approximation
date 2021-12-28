import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from .callback import Callback
from .base import BaseLogger


class SummaryWriter(SummaryWriter):

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


class TensorBoardLogger(BaseLogger, Callback):

    def on_train_epoch_end(self, model):
        for metric in model.metrics["train"]:
            self.experiment.add_scalar(
                f"train/{metric}", model.metrics["train"][metric], model.current_epoch
            )

    def on_valid_epoch_end(self, model):
        for metric in model.metrics["valid"]:
            self.experiment.add_scalar(
                f"valid/{metric}", model.metrics["valid"][metric], model.current_epoch
            )
