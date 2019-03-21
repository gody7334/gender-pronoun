import os
import random
import logging
from pathlib import Path
from collections import deque
from matplotlib.pyplot import *
from pprint import pprint as pp

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from utils.lr_finder import LRFinder
from utils.project import Global as G
from utils.lr_scheduler import TriangularLR

AVERAGING_WINDOW = 300
SEED = int(os.environ.get("SEED", 9293))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class OneCycle:
    """
    run one cycle policy,
    advantage of this policy is,
    it can reset many paramenter to fine tune model,
    epoch: will be converted into n_step
    TODO diffrentiate learning rate
    """

    def __init__(self, bot, *args, **kwargs):
        self.bot = bot
        self.n_step = None
        self.n_epoch = None
        self.steps_per_epoch = None

    def update_bot(self,
        train_loader=None, val_loader=None,
        optimizer=None, criterion=None, scheduler=None,
        pretrained_path='', unfreeze_layers=[], unfreeze_all=False,
        lrs=[], n_epoch=None, n_step=None, stage='',
        accu_gradient_step=None):

        assert len(unfreeze_layers) != 0 or unfreeze_all is True, "no layers will be trained"
        assert n_epoch is not None or n_step is not None, "need to assign train step"
        self.n_step = n_step
        self.n_epoch = n_epoch

        if pretrained_path != '': self.bot.load_model(pretrained_path);
        if train_loader is not None: self.bot.train_loader=train_loader; print('reset train_loader: '+ str(train_loader));
        if val_loader is not None: self.bot.val_loader=val_loader; print('reset val_loader: '+ str(val_loader));
        if optimizer is not None: self.bot.optimizer=optimizer; print('reset optimizer: '+ str(optimizer)+' lr, etc.. will also be reset');
        if criterion is not None: self.bot.criterion=criterion; print('reset criterion: '+ str(criterion));
        if stage is not None: self.bot.stage = stage;
        if accu_gradient_step is not None: self.bot.accu_gradient_step = accu_gradient_step;

        self.steps_per_epoch = len(self.bot.train_loader)
        if n_step is None: self.n_step = self.steps_per_epoch*n_epoch;

        '''
        one cycle default scheduler,
        one cycle(len of one cycle) is decided by n_step
        or you can assign customized scheduler
        '''
        if scheduler is not None:
            if scheduler == 'Default Triangular':
                self.bot.scheduler=TriangularLR(self.bot.optimizer, 11, ratio=3, steps_per_cycle=self.n_step)
            else:
                self.bot.scheduler=scheduler; print('reset scheduler: '+ str(scheduler));

        if unfreeze_all is False:
            for param in self.bot.model.parameters():
                param.requires_grad = False
            self.bot.set_trainable(unfreeze_layers, True)
        else:
            for param in self.bot.model.parameters():
                param.requires_grad = True

    def train_one_cycle(self):
        assert self.n_step is not None, "need to assign train step"

        self.bot.train(
            self.n_step,
            log_interval=self.steps_per_epoch // 20,
            eval_interval=self.steps_per_epoch
            )

class BaseBot:
    """
    Base Interface to Model Training and Inference
    snapshot_policy:
        'validate': save every validation
        'best': save best, (in the future... its complicate as need monitor metrics)
        'last': save only last
    accu_gradient_step: steps to accumulate gradient,
        beaware some NN not compatible to this trick,
        ex: batch normalization, which required statistic accrss batch
        If using accumulated gradient, those layer have to be freezed, or customized implement.
    """

    name = "basebot"

    def __init__(
            self, model, train_loader, val_loader, optimizer, criterion,
            scheduler=None, clip_grad=0, avg_window=AVERAGING_WINDOW,
            batch_idx=0,echo=False, device="cuda:0", use_tensorboard=False,
            snapshot_policy='validate', stage='0', accu_gradient_step=1):

        assert model is not None
        assert optimizer is not None
        assert criterion is not None
        assert train_loader is not None
        assert val_loader is not None
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler

        self.device = device
        self.avg_window = avg_window
        self.clip_grad = clip_grad
        self.batch_idx = batch_idx
        self.snapshot_policy = snapshot_policy
        self.stage=stage
        self.accu_gradient_step=accu_gradient_step

        self.logger = G.logger
        self.logger.info("SEED: %s", SEED)
        self.checkpoint_dir = Path(G.proj.check_point)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.loss_format = "%.8f"

        self.best_performers = []
        self.step = 0
        self.train_losses = None
        self.train_weights = None
        self.eval_losses = None
        self.eval_weights = None

        self.count_model_parameters()

    def extract_prediction(self, output):
        """Assumes single output"""
        # return output
        return output[:,0]

    def transform_prediction(self, prediction):
        '''
        override if needed, ex: topN acc,
        '''
        return prediction

    def metrics(self, outputs, targets):
        '''
        override if needed for different metrics
        '''
        criterion_scores = self.criterion(outputs, targets).data.cpu().numpy()
        score = np.mean(criterion_scores)
        return score

    def train_one_step(self, input_tensors, target):
        '''
        override if needed
        ex: multiple criterion, multiple model output
        '''
        self.model.train()
        assert self.model.training

        if self.step % self.accu_gradient_step == 0:
            self.optimizer.zero_grad()

        output = self.model(*input_tensors)
        batch_loss = self.criterion(self.extract_prediction(output), target)

        # devide batch size to make train more stable,
        # avg loss wont effected by batch size, so lr,
        avg_batch_loss = batch_loss/len(target)
        avg_batch_loss.backward()

        self.train_losses.append(batch_loss.data.cpu().numpy())
        self.train_weights.append(target.size(self.batch_idx))

        if self.step % self.accu_gradient_step == 0:
            if self.clip_grad > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

    def eval_one_step(self, input_tensors, y_local):
        '''
        override if needed
        ex: multiple criterion, multiple model output
        '''
        self.eval_losses = []
        self.eval_weights = []
        input_tensors = [x.to(self.device) for x in input_tensors]
        output = self.model(*input_tensors)
        batch_loss = self.criterion(
            self.extract_prediction(output), y_local.to(self.device))
        self.eval_losses.append(batch_loss.data.cpu().numpy())
        self.eval_weights.append(y_local.size(self.batch_idx))

    def snapshot(self, loss, rule='min'):
        #TODO save every validate or save best,
        if self.snapshot_policy == 'valid' or \
            self.snapshot_policy == 'last':
            loss_str = self.loss_format % loss
            target_path = (
                self.checkpoint_dir /
                "stage{}_snapshot_{}_{}.pth".format(self.stage, self.name, loss_str))
            self.best_performers.append((loss, target_path, self.step))
            self.logger.info("Saving checkpoint %s...", target_path)
            torch.save(self.model.state_dict(), target_path)
            assert Path(target_path).exists()

    def set_trainable(self, l, b):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())

        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b

        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)

        for g in l:
            apply_leaf(g, lambda m: set_trainable_attr(m, b))

    def lr_finder(self, end_lr=10, num_iter=100, img_path='./'):
        print('Start finding LR')
        lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device=("cuda:0"))
        lr_finder.range_test(self.train_loader, end_lr=end_lr, num_iter=num_iter)
        lr_finder.plot()
        savefig(img_path)
        clf()

    def count_model_parameters(self):
        self.logger.info(
            "# of paramters: {:,d}".format(
                np.sum(p.numel() for p in self.model.parameters())))
        self.logger.info(
            "# of trainable paramters: {:,d}".format(
                np.sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def log_progress(self):
        train_loss_avg = np.average(
            self.train_losses, weights=self.train_weights)
        self.logger.info(
            "Step %s: train %.6f lr: %.3e",
            self.step, train_loss_avg, self.optimizer.param_groups[-1]['lr'])
        self.logger.tb_scalars(
                "lr", {"lr": self.optimizer.param_groups[0]['lr']}, self.step)
        self.logger.tb_scalars(
            "losses", {"train": train_loss_avg}, self.step)

    def train(
            self, n_steps, *, log_interval=50,
            early_stopping_cnt=0, min_improv=1e-4,
            eval_interval=2500):
        self.train_losses = deque(maxlen=self.avg_window)
        self.train_weights = deque(maxlen=self.avg_window)

        if self.val_loader is not None:
            best_val_loss = 100
        epoch = 0
        local_step = 0
        wo_improvement = 0
        self.best_performers = []
        self.logger.info(
            "Optimizer {}".format(str(self.optimizer)))
        self.logger.info("Batches per epoch: {}".format(
            len(self.train_loader)))
        try:
            while local_step < n_steps:
                epoch += 1
                self.logger.info(
                    "=" * 20 + "Epoch %d" + "=" * 20, epoch)

                # one epoch
                for *input_tensors, target in self.train_loader:
                    input_tensors = [x.to(self.device) for x in input_tensors]

                    # train on batch
                    self.train_one_step(input_tensors, target.to(self.device))
                    self.step += 1
                    local_step += 1

                    # train log
                    if self.step % log_interval == 0:
                        self.log_progress()

                    # eval
                    if self.step % eval_interval == 0:
                        loss = self.eval(self.val_loader)

                        if self.snapshot_policy is 'validate' or \
                            self.snapshot_policy is 'best':
                            self.snapshot(loss)

                        if best_val_loss > loss + min_improv:
                            self.logger.info("New low\n")
                            best_val_loss = loss
                            wo_improvement = 0
                        else:
                            wo_improvement += 1

                    if self.scheduler:
                        self.scheduler.step()
                    if early_stopping_cnt and wo_improvement > early_stopping_cnt:
                        return
                    if local_step >= n_steps:
                        break

            # save only last
            if self.snapshot_policy is 'last':
                loss = self.eval(self.val_loader)
                self.snapshot(loss)

        except KeyboardInterrupt:
            pass
        self.best_performers = sorted(self.best_performers, key=lambda x: x[0])
        self.logger.tb_export_scalars()

    def eval(self, loader):
        self.model.eval()
        losses, weights = [], []
        with torch.set_grad_enabled(False):
            for *input_tensors, y_local in tqdm(loader):
                self.eval_one_step(input_tensors, y_local)
        loss = np.average(self.eval_losses, weights=self.eval_weights)
        loss_str = self.loss_format % loss
        self.logger.info("Snapshot loss %s", loss_str)
        self.logger.tb_scalars(
            "losses", {"val": loss},  self.step)

        return loss

    def predict_batch(self, input_tensors):
        self.model.eval()
        tmp = self.model(*input_tensors)
        return self.extract_prediction(tmp)

    def predict_avg(self, loader, k=8):
        '''
        avg ensemble
        '''
        assert len(self.best_performers) >= k
        preds = []
        # Iterating through checkpoints
        for i in range(k):
            target = self.best_performers[i][1]
            self.logger.info("Loading %s", format(target))
            self.load_model(target)
            preds.append(self.predict(loader).unsqueeze(0))
        return torch.cat(preds, dim=0).mean(dim=0)

    def predict(self, loader, *, return_y=False):
        '''
        test set has label which can be used to investigate manually
        '''
        self.model.eval()
        outputs, y_global = [], []
        with torch.set_grad_enabled(False):
            for *input_tensors, y_local in tqdm(loader):
                input_tensors = [x.to(self.device) for x in input_tensors]
                outputs.append(self.predict_batch(input_tensors).cpu())
                y_global.append(y_local.cpu())
            outputs = torch.cat(outputs, dim=0)
            y_global = torch.cat(y_global, dim=0)
        if return_y:
            return outputs, y_global
        return outputs

    def remove_checkpoints(self, keep=0):
        for checkpoint in np.unique([x[1] for x in self.best_performers[keep:]]):
            Path(checkpoint).unlink()
        self.best_performers = self.best_performers[:keep]

    def load_model(self, target_path):
        self.model.load_state_dict(torch.load(target_path))
