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
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.batchnorm import _BatchNorm
from pytorch_pretrained_bert.modeling import BertModel, BertLayerNorm

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
        self.stage_params = \
        [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-5,weight_decay=1e-3),
                    'batch_size': [2,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },
        ]

    def do_cycles_train(self):
        stage=0
        while(stage<len(self.stage_params)):
            params = self.stage_params[stage]
            G.logger.info("Start stage %s", str(stage))

            if params['batch_size'] is not None:
                self.gapdl.update_batch_size(
                        train_size=params['batch_size'][0],
                        val_size=params['batch_size'][1],
                        test_size=params['batch_size'][2])

            self.oc.update_bot(optimizer = params['optimizer'],
                    scheduler=params['scheduler'],
                    unfreeze_layers=params['unfreeze_layers'],
                    freeze_layers=params['freeze_layers'],
                    n_epoch=params['epoch'],
                    stage=str(stage),
                    train_loader=self.gapdl.train_loader,
                    val_loader=self.gapdl.val_loader,
                    )
            self.oc.train_one_cycle()
            stage+=1

            if mode=="DEV" and stage==5:
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
    gappl.do_cycles_train()
    # score = gappl.do_prediction()

    G.logger.info('success!')
