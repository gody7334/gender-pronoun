import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from utils.bot import BaseBot
from utils.project import Global as G


class GAPBot(BaseBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_format = "%.6f"

    def extract_prediction(self, tensor):
        return tensor

    def predict_avg(self, loader, checkpoint_path, regex_pattern=''):
        '''
        avg ensemble
        '''
        preds = []

        # Iterating through checkpoints
        for i in range(k):
            target = self.best_performers[i][1]
            self.logger.info("Loading %s", format(target))
            self.load_model(target)
            preds.append(self.predict(loader).unsqueeze(0))
        return torch.cat(preds, dim=0).mean(dim=0)


    # def metrics(self, outputs, targets):
        # '''
        # override if needed for different metrics
        # '''
        # criterion_scores = self.criterion(outputs, targets).data.cpu().numpy()
        # score = np.mean(criterion_scores)
        # G.logger.info("holdout validation score: %.6f", score)
        # G.logger.tb_scalars("losses", {"Holdout": score}, self.step)

        # for t in np.arange(0.9,1.0,0.01):
            # import ipdb; ipdb.set_trace();
            # outputs_sm = nn.functional.softmax(outputs,dim=1)
            # outputs_t_idx = torch.sum((outputs_sm>t).float()*1,dim=1).unsqueeze(1)

            # outputs_t = ((outputs_sm > t).float() * 0.999) + ((outputs_sm <= t).float() * 0.0005)+1e-8
            # outputs_t = outputs_t * outputs_t_idx + outputs_sm * (1-outputs_t_idx)

            # outputs_t = torch.log(outputs_t)
            # loss = nn.NLLLoss()
            # criterion_scores = loss(outputs_t, targets).data.cpu().numpy()
            # score = np.mean(criterion_scores)
            # G.logger.info("threshold: %.2f, holdout validation score: %.6f", t, score)

        # return score


    ## keep best
    # def snapshot(self, loss):
        # """Override the snapshot method because Kaggle kernel has limited local disk space."""
        # loss_str = self.loss_format % loss
        # self.logger.info("Snapshot loss %s", loss_str)
        # self.logger.tb_scalars(
            # "losses", {"val": loss},  self.step)
        # target_path =(
            # self.checkpoint_dir /
            # "snapshot_{}_{}.pth".format(self.name, loss_str))

        # if not self.best_performers or (self.best_performers[0][0] > loss) or self.snapshot_policy=='last':
            # torch.save(self.model.state_dict(), target_path)
            # self.best_performers = [(loss, target_path, self.step)]
        # self.logger.info("Saving checkpoint %s...", target_path)
        # assert Path(target_path).exists()
        # return loss
