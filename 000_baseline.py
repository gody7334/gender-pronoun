############ Setup Project ######################
import os
import matplotlib
from utils.project import Global as G

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
matplotlib.use('Agg')

EXPERIMENT="000_BASELINE"
DISCRIPTION='create baseline'
gpu_id = '0'
G(EXPERIMENT, DISCRIPTION, gpu_id)

mode="EXP"

##################################################

import logging
from pathlib import Path
import colored_traceback.always
from matplotlib.pyplot import *

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from utils.bot import OneCycle
from utils.lr_finder import LRFinder
from utils.lr_scheduler import TriangularLR

from dev.gap_bot import *
from dev.data import *
from dev.model import *

####################################################

class GAPPipeline:
    def __init__(self):
        G.logger.info("load model")
        self.model = GAPModel(BERT_MODEL, torch.device("cuda:0"))

        G.logger.info("load gapdl")
        self.gapdl = GAPDataLoader()

        G.logger.info("create bot")
        self.bot = GAPBot(
            self.model, self.gapdl.train_loader, self.gapdl.val_loader,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3),
            criterion=torch.nn.CrossEntropyLoss(),
            echo=True,
            use_tensorboard=True,
            avg_window=25,
            snapshot_policy='last'
        )

        G.logger.info("create onecycle")
        self.oc = OneCycle(self.bot)
        self.set_cycles_train_params()

    def set_cycles_train_params(self):
        self.stage_optimizer = [
                Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-3),
                Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-3),
                Adam(self.model.parameters(),lr=1e-5,weight_decay=1e-3),
                ]
        self.stage_batch_size = [
                None,
                None,
                None,
                None,
                [8,128,128]
                ]
        self.stage_scheduler = [
                "Default Triangular",
                "Default Triangular",
                "Default Triangular",
                "Default Triangular",
                "Default Triangular",
                ]
        self.stage_unfreeze_layers = [
                [self.model.head],
                [self.model.head],
                [self.model.head],
                [self.model.head],
                [self.model.head, self.model.bert.encoder]
                ]

        if mode=="EXP":
            self.stage_epoch = [5,10,20,20,20]
        elif mode=="DEV":
            self.stage_epoch = [1,1,1,1,1]

    def do_cycles_train(self):
        stage=0
        while(stage<len(self.stage_epoch)):
            G.logger.info("Start stage %s", str(stage))

            if self.stage_batch_size[stage] is not None:
                self.gapdl.update_batch_size(
                        train_size=self.stage_batch_size[stage][0],
                        val_size=self.stage_batch_size[stage][1],
                        test_size=self.stage_batch_size[stage][2])

            self.oc.update_bot(optimizer = self.stage_optimizer[stage],
                    scheduler=self.stage_scheduler[stage],
                    unfreeze_layers=self.stage_unfreeze_layers[stage],
                    n_epoch=self.stage_epoch[stage],
                    stage=str(stage),
                    train_loader=self.gapdl.train_loader,
                    val_loader=self.gapdl.val_loader,
                    )
            self.oc.train_one_cycle()
            stage+=1

            if mode=="DEV" and stage==3:
                break

    def do_cycles_train_old(self):
        'stage1'
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        oc.update_bot(optimizer=optimizer,
                scheduler="Default Triangular",
                unfreeze_layers=[model.head],
                n_epoch=10,
                stage='1',
                )
        oc.train_one_cycle()

        'stage2'
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
        oc.update_bot(optimizer = optimizer,
                scheduler="Default Triangular",
                unfreeze_layers=[model.head],
                n_epoch=20,
                stage='2',
                )
        oc.train_one_cycle()

        'stage3'
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
        oc.update_bot(optimizer = optimizer,
                scheduler="Default Triangular",
                unfreeze_layers=[model.head],
                n_epoch=20,
                stage='3',
                )
        oc.train_one_cycle()


        'stage4'
        gapdl.update_batch_size(train_size=8,val_size=128,test_size=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-2)
        oc.update_bot(optimizer = optimizer,
                scheduler="Default Triangular",
                train_loader=gapdl.train_loader,
                val_loader=gapdl.val_loader,
                unfreeze_layers=[model.head, model.bert.encoder],
                n_epoch=20,
                stage='4',
                )
        oc.train_one_cycle()

    def do_prediction(self):
        target_path = '/home/gody7334/gender-pronoun/input/result/000_BASELINE/2019-03-21_00-27-14/check_point/stage4_snapshot_basebot_0.506462.pth'
        self.bot.load_model(target_path)
        outputs, targets =  self.bot.predict(self.gapdl.test_loader, return_y=True)
        score = self.bot.metrics(outputs, targets)
        import ipdb; ipdb.set_trace();
        return score


if __name__ == '__main__':
    G.logger.info( '%s: calling main function ... ' % os.path.basename(__file__))

    gappl = GAPPipeline()
    # gappl.do_cycles_train()
    score = gappl.do_prediction()

    G.logger.info('success!')
