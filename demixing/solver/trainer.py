import psutil
import torch
import torch.nn as nn
from enum import Enum
from tqdm.auto import tqdm
from typing import List
from abc import abstractmethod


class Module(nn.Module):

    def __init__(self, *args, **kwargs):
        """
        Instead of inheriting from nn.Module, you import ccsa and inherit from ccsa.Module
        """
        super().__init__(*args, **kwargs)
        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.scheduler = None
        self.step_scheduler_after = None
        self.step_scheduler_metric = None
        self.current_epoch = 0
        self.current_train_step = 0
        self.current_valid_step = 0
        self._model_state = None
        self._train_state = None
        self.device = None
        self._callback_runner = None
        self.fp16 = False
        self.scaler = None
        self.accumulation_steps = 0
        self.batch_index = 0
        self.metrics = {}
        self.metrics["train"] = {}
        self.metrics["valid"] = {}
        self.metrics["test"] = {}
        self.clip_grad_norm = None

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value
        # run something here in future if needed

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, value):
        self._train_state = value
        if self._callback_runner is not None:
            self._callback_runner(value)

    def name_to_metric(self, metric_name):
        if metric_name == "current_epoch":
            return self.current_epoch
        v_1 = metric_name.split("_")[0]
        v_2 = "_".join(metric_name.split("_")[1:])
        return self.metrics[v_1][v_2]

    def _init_model(
        self,
        device,
        train_dataset,
        valid_dataset,
        train_sampler,
        valid_sampler,
        train_bs,
        valid_bs,
        n_jobs,
        callbacks,
        fp16,
        train_collate_fn,
        valid_collate_fn,
        train_shuffle,
        valid_shuffle,
        accumulation_steps,
        clip_grad_norm,
    ):

        if callbacks is None:
            callbacks = list()

        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        self.device = device
        self.accumulation_steps = accumulation_steps
        self.clip_grad_norm = clip_grad_norm

        if next(self.parameters()).device != self.device:
            self.to(self.device)

        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_bs,
                num_workers=n_jobs,
                sampler=train_sampler,
                shuffle=train_shuffle,
                collate_fn=train_collate_fn,
            )
        if self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=valid_bs,
                    num_workers=n_jobs,
                    sampler=valid_sampler,
                    shuffle=valid_shuffle,
                    collate_fn=valid_collate_fn,
                )

        if self.optimizer is None:
            self.optimizer = self.fetch_optimizer()

        if self.scheduler is None:
            self.scheduler = self.fetch_scheduler()

        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self._callback_runner = CallbackRunner(callbacks, self)
        self.train_state = TrainingState.TRAIN_START

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        return {fn.__name__:fn(targets, outputs) for fn in self.metrics_fn}

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return self.loss_fn(outputs, targets)

    def fetch_optimizer(self, *args, **kwargs):
        return

    def fetch_scheduler(self, *args, **kwargs):
        return

    @abstractmethod
    def compile(self, loss_fn, optimizer, metrics_fn, scheduler=None):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics_fn = metrics_fn

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def model_fn(self, data):
        for key, value in data.items():
            data[key] = value.to(self.device)
        if self.fp16:
            with torch.cuda.amp.autocast():
                output, loss, metrics = self(**data)
        else:
            output, loss, metrics = self(**data)
        return output, loss, metrics

    def train_one_step(self, data):
        if self.accumulation_steps == 1 and self.batch_index == 0:
            self.zero_grad()
        _, loss, metrics = self.model_fn(data)
        loss = loss / self.accumulation_steps
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        if (self.batch_index + 1) % self.accumulation_steps == 0:
            if self.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler:
                if self.step_scheduler_after == "batch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            if self.batch_index > 0:
                self.zero_grad()
        return loss, metrics

    def validate_one_step(self, data):
        _, loss, metrics = self.model_fn(data)
        return loss, metrics

    def predict_one_step(self, data):
        output, _, _ = self.model_fn(data)
        return output

    def update_metrics(self, losses, monitor):
        self.metrics[self._model_state.value].update(monitor)
        self.metrics[self._model_state.value]["loss"] = losses.avg

    def train_one_epoch(self, data_loader):
        self.train()
        self.model_state = ModelState.TRAIN
        losses = AverageMeter()
        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.batch_index = b_idx
            self.train_state = TrainingState.TRAIN_STEP_START
            loss, metrics = self.train_one_step(data)
            self.train_state = TrainingState.TRAIN_STEP_END
            losses.update(loss.item(), data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            self.current_train_step += 1
            tk0.set_postfix(loss=losses.avg, stage="train", **monitor)
        tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)

        return losses.avg

    def validate_one_epoch(self, data_loader):
        self.eval()
        self.model_state = ModelState.VALID
        losses = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.train_state = TrainingState.VALID_STEP_START
            with torch.no_grad():
                loss, metrics = self.validate_one_step(data)
            self.train_state = TrainingState.VALID_STEP_END
            losses.update(loss.item(), data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            tk0.set_postfix(loss=losses.avg, stage="valid", **monitor)
            self.current_valid_step += 1
        tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg

    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output

    def predict(self, dataset, sampler=None, batch_size=16, n_jobs=1, collate_fn=None):
        if next(self.parameters()).device != self.device:
            self.to(self.device)

        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        if batch_size == 1:
            n_jobs = 0
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=n_jobs, sampler=sampler, collate_fn=collate_fn, pin_memory=True
        )

        if self.training:
            self.eval()

        else:
            tk0 = tqdm(data_loader, total=len(data_loader))

        for _, data in enumerate(tk0):
            with torch.no_grad():
                out = self.predict_one_step(data)
                out = self.process_output(out)
                yield out

            tk0.set_postfix(stage="test")

        tk0.close()

    def save(self, model_path, weights_only=False):
        model_state_dict = self.state_dict()
        if weights_only:
            torch.save(model_state_dict, model_path)
            return
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict["epoch"] = self.current_epoch
        model_dict["fp16"] = self.fp16
        torch.save(model_dict, model_path)

    def load(self, model_path, weights_only=False, device="cuda"):
        self.device = device
        if next(self.parameters()).device != self.device:
            self.to(self.device)
        model_dict = torch.load(model_path, map_location=torch.device(device))
        if weights_only:
            self.load_state_dict(model_dict)
        else:
            self.load_state_dict(model_dict["state_dict"])

    def fit(
        self,
        train_dataset,
        valid_dataset=None,
        train_sampler=None,
        valid_sampler=None,
        device="cuda",
        epochs=10,
        train_bs=16,
        valid_bs=16,
        n_jobs=8,
        callbacks=None,
        fp16=False,
        train_collate_fn=None,
        valid_collate_fn=None,
        train_shuffle=True,
        valid_shuffle=False,
        accumulation_steps=1,
        clip_grad_norm=None,
    ):
        self._init_model(
            device=device,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_sampler=train_sampler,
            valid_sampler=valid_sampler,
            train_bs=train_bs,
            valid_bs=valid_bs,
            n_jobs=n_jobs,
            callbacks=callbacks,
            fp16=fp16,
            train_collate_fn=train_collate_fn,
            valid_collate_fn=valid_collate_fn,
            train_shuffle=train_shuffle,
            valid_shuffle=valid_shuffle,
            accumulation_steps=accumulation_steps,
            clip_grad_norm=clip_grad_norm,
        )

        for _ in range(epochs):
            self.train_state = TrainingState.EPOCH_START
            self.train_state = TrainingState.TRAIN_EPOCH_START
            train_loss = self.train_one_epoch(self.train_loader)
            self.train_state = TrainingState.TRAIN_EPOCH_END
            if self.valid_loader:
                self.train_state = TrainingState.VALID_EPOCH_START
                valid_loss = self.validate_one_epoch(self.valid_loader)
                self.train_state = TrainingState.VALID_EPOCH_END
            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            self.train_state = TrainingState.EPOCH_END
            if self._model_state.value == "end":
                break
            self.current_epoch += 1
        self.train_state = TrainingState.TRAIN_END


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelState(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    END = "end"


class TrainingState(Enum):
    TRAIN_START = "on_train_start"
    TRAIN_END = "on_train_end"
    EPOCH_START = "on_epoch_start"
    EPOCH_END = "on_epoch_end"
    TRAIN_EPOCH_START = "on_train_epoch_start"
    TRAIN_EPOCH_END = "on_train_epoch_end"
    VALID_EPOCH_START = "on_valid_epoch_start"
    VALID_EPOCH_END = "on_valid_epoch_end"
    TRAIN_STEP_START = "on_train_step_start"
    TRAIN_STEP_END = "on_train_step_end"
    VALID_STEP_START = "on_valid_step_start"
    VALID_STEP_END = "on_valid_step_end"
    TEST_STEP_START = "on_test_step_start"
    TEST_STEP_END = "on_test_step_end"


class Callback:

    def on_epoch_start(self, model, **kwargs):
        return

    def on_epoch_end(self, model, **kwargs):
        return

    def on_train_epoch_start(self, model, **kwargs):
        return

    def on_train_epoch_end(self, model, **kwargs):
        return

    def on_valid_epoch_start(self, model, **kwargs):
        return

    def on_valid_epoch_end(self, model, **kwargs):
        return

    def on_train_step_start(self, model, **kwargs):
        return

    def on_train_step_end(self, model, **kwargs):
        return

    def on_valid_step_start(self, model, **kwargs):
        return

    def on_valid_step_end(self, model, **kwargs):
        return

    def on_test_step_start(self, model, **kwargs):
        return

    def on_test_step_end(self, model, **kwargs):
        return

    def on_train_start(self, model, **kwargs):
        return

    def on_train_end(self, model, **kwargs):
        return


class CallbackRunner:

    def __init__(self, callbacks: List[Callback], model):
        self.model = model
        self.callbacks = callbacks

    def __call__(self, current_state, **kwargs):
        for cb in self.callbacks:
            _ = getattr(cb, current_state.value)(self.model, **kwargs)