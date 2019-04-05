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
from pprint import pprint as pp

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.lr_scheduler import StepLR
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
        '''
        Gradient Checkpoint model need turn off Dropout, and BatchNorm
        Accumulated Gradient need turn off BatchNorm
        '''
        G.logger.info("load model")
        # self.model = GAPModel_CheckPoint(BERT_MODEL, torch.device("cuda:0"))
        # self.model = GAPModel(BERT_MODEL, torch.device("cuda:0"))
        self.model = score_model(BERT_MODEL)

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
        # load pretained model for continue training
        self.oc.update_bot(pretrained_path='', continue_step=0, n_step=0)

        self.stage_params = PipelineParams(self.model).baseline()
        # flatten list list dict to list dict
        self.stage_params = [j for sub in self.stage_params for j in sub]

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
                    dropout_ratio=params['dropout_ratio'],
                    n_epoch=params['epoch'],
                    stage=str(stage),
                    train_loader=self.gapdl.train_loader,
                    val_loader=self.gapdl.val_loader,
                    )
            self.oc.train_one_cycle()
            self.do_prediction('')
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

    def do_prediction(self, target_path=''):
        if target_path != '':
            self.bot.load_model(target_path)

        outputs, targets =  self.bot.predict(self.gapdl.test_loader, return_y=True)
        self.bot.metrics(outputs, targets)

class PipelineParams():
    def __init__(self,model):
        self.model = model
        self.params = []
        pass

    def step_scheduler(self):
        '''
        simple step schedulre as baseline
        '''
        adam = Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4)
        self.params = \
            [
                {
                    'optimizer': adam,
                    'batch_size': [20,128,128],
                    'scheduler': StepLR(adam, 1000, gamma=0.5, last_epoch=-1),
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 50 if mode=="EXP" else 1,
                },
            ]
        return self.params

    def increase_dropout(self):
        '''
        remove BERT dropout, if don't train BERT
        slowly decrease dropout ratio in HEAD when finetune
        maybe final finetune...
        '''
        self.params = \
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [40,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [(self.model.bert, 0.0)],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4),
                    'batch_size': [40,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [(self.model.bert, 0.0),
                                      (self.model.head, 0.60)],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4),
                    'batch_size': [40,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [(self.model.bert, 0.00),
                                      (self.model.head, 0.60)],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-5,weight_decay=1e-4),
                    'batch_size': [40,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio':  [(self.model.bert, 0.00),
                                      (self.model.head, 0.60)],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-5,weight_decay=1e-4),
                    'batch_size': [40,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio':  [(self.model.bert, 0.0),
                                      (self.model.head, 0.7)],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
            ]
        return self.params

    def baseline(self):
        '''
        baseline, one cycle train, with reducing lr after one cycle
        '''
        self.params = \
            [
                [{
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                }]*1,
                [{
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }]*3,
                [{
                    'optimizer': Adam(self.model.parameters(),lr=5e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }]*3,
                [{
                    'optimizer': Adam(self.model.parameters(),lr=2.5e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.embedding, nn.Module),
                                        (self.model.span_extractor, nn.Module),
                                        (self.model.pair_score, nn.Module),
                                        ],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                }]*3,
            ]
        return self.params

    def accumulated_gradient(self):
        '''
        warm up head and init BN without accu_gradient
        then turn off BN train and using accu_gradient to reduce var
        without training BERT
        '''
        self.params = \
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert, nn.Module),
                                      (self.model.head, _BatchNorm)],
                    'dropout_ratio': [],
                    'accu_gradient_step': 10,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert, nn.Module),
                                      (self.model.head, _BatchNorm)],
                    'dropout_ratio': [],
                    'accu_gradient_step': 10,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert, nn.Module),
                                      (self.model.head, _BatchNorm)],
                    'dropout_ratio': [],
                    'accu_gradient_step': 10,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-5,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert, nn.Module),
                                      (self.model.head, _BatchNorm)],
                    'dropout_ratio': [],
                    'accu_gradient_step': 10,
                    'epoch': 20 if mode=="EXP" else 1,
                },
            ]
        return self.params

    def unfreeze_bert(self):
        '''
        inital warm up training head,
        then unfreeze bert to train all model
        if using gradient checkpoint, need to turn off dropout
        if using accumulated gradient, need to turn off BN
        '''
        self.params = \
            [
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 2 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-4},
                         {'params':self.model.bert.parameters(),'lr':1e-5},],
                        weight_decay=1e-3),
                    'batch_size': [6,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module)],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-5},
                         {'params':self.model.bert.parameters(),'lr':1e-6},],
                        weight_decay=1e-3),
                    'batch_size': [6,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                # {
                    # 'optimizer': Adam(
                        # [{'params':self.model.head.parameters(),'lr':1e-4},
                         # {'params':self.model.bert.parameters(),'lr':1e-5},],
                        # weight_decay=1e-3),
                    # 'batch_size': [6,128,128],
                    # 'scheduler': "Default Triangular",
                    # 'unfreeze_layers': [],
                    # 'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    # 'dropout_ratio': [],
                    # 'accu_gradient_step': None,
                    # 'epoch': 20 if mode=="EXP" else 1,
                # },
                # {
                    # 'optimizer': Adam(
                        # [{'params':self.model.head.parameters(),'lr':1e-5},
                         # {'params':self.model.bert.parameters(),'lr':1e-6},],
                        # weight_decay=1e-3),
                    # 'batch_size': [6,128,128],
                    # 'scheduler': "Default Triangular",
                    # 'unfreeze_layers': [],
                    # 'dropout_ratio': [],
                    # 'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    # 'accu_gradient_step': None,
                    # 'epoch': 20 if mode=="EXP" else 1,
                # },
            ]
        return self.params

    def unfreeze_bert_with_accu_gradient(self):
        '''
        !!!! do not inital warm up training head,
        !!!! it will cause fail to train..
        then unfreeze bert to train all model
        freeze batch norm, bert layer norm
        for using accu gradient to reduce variance as batch size is too small
        BERT is very sensitive, need to train under very small LR
        add dropout adjustment, as previous have huge performance gap
        '''
        self.params = \
            [
                # {
                    # 'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-3),
                    # 'batch_size': [40,128,128],
                    # 'scheduler': "Default Triangular",
                    # 'unfreeze_layers': [(self.model.head, nn.Module)],
                    # 'freeze_layers': [],
                    # 'dropout_ratio': [],
                    # 'accu_gradient_step': None,
                    # 'epoch': 5 if mode=="EXP" else 1,
                # },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-3},
                         {'params':self.model.bert.parameters(),'lr':1e-4},],
                        weight_decay=1e-3),
                    'batch_size': [4,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),
                                      (self.model,(_BatchNorm, BertLayerNorm))],
                    'dropout_ratio': [],
                    'accu_gradient_step': 100,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-3},
                         {'params':self.model.bert.parameters(),'lr':1e-4},],
                        weight_decay=1e-3),
                    'batch_size': [4,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),
                                      (self.model,(_BatchNorm, BertLayerNorm))],
                    'dropout_ratio': [],
                    'accu_gradient_step': 100,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-4},
                         {'params':self.model.bert.parameters(),'lr':1e-5},],
                        weight_decay=1e-3),
                    'batch_size': [4,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),
                                      (self.model,(_BatchNorm, BertLayerNorm))],
                    'dropout_ratio': [],
                    'accu_gradient_step': 100,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-4},
                         {'params':self.model.bert.parameters(),'lr':1e-5},],
                        weight_decay=1e-3),
                    'batch_size': [4,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),
                                      (self.model,(_BatchNorm,BertLayerNorm))],
                    'dropout_ratio': [],
                    'accu_gradient_step': 100,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-5},
                         {'params':self.model.bert.parameters(),'lr':1e-6},],
                        weight_decay=1e-3),
                    'batch_size': [4,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),
                                      (self.model,(_BatchNorm,BertLayerNorm))],
                    'dropout_ratio': [],
                    'accu_gradient_step': 100,
                    'epoch': 20 if mode=="EXP" else 1,
                },

            ]
        return self.params

    def finetune_bert(self):
        '''
        one cycle train as baseline,
        only fine tune whole model in last cycle
        its better not to turn off dropout, as it will cause overfitting
        '''
        self.params = \
            [
               {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 5 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 10 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4),
                    'batch_size': [20,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [(self.model.head, nn.Module)],
                    'freeze_layers': [],
                    'dropout_ratio': [],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-5},
                         {'params':self.model.bert.parameters(),'lr':1e-6},],
                        weight_decay=1e-4),
                    'batch_size': [6,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'dropout_ratio': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-5},
                         {'params':self.model.bert.parameters(),'lr':1e-6},],
                        weight_decay=1e-4),
                    'batch_size': [6,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'dropout_ratio': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-6},
                         {'params':self.model.bert.parameters(),'lr':1e-7},],
                        weight_decay=1e-4),
                    'batch_size': [6,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'dropout_ratio': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },
                {
                    'optimizer': Adam(
                        [{'params':self.model.head.parameters(),'lr':1e-6},
                         {'params':self.model.bert.parameters(),'lr':1e-7},],
                        weight_decay=1e-4),
                    'batch_size': [6,128,128],
                    'scheduler': "Default Triangular",
                    'unfreeze_layers': [],
                    'dropout_ratio': [],
                    'freeze_layers': [(self.model.bert.embeddings,nn.Module),],
                    'accu_gradient_step': None,
                    'epoch': 20 if mode=="EXP" else 1,
                },


            ]
        return self.params


if __name__ == '__main__':
    G.logger.info( '%s: calling main function ... ' % os.path.basename(__file__))

    gappl = GAPPipeline()
    gappl.do_cycles_train()

    # target_path = '/home/gody7334/gender-pronoun/input/result/000_BASELINE/2019-03-21_00-27-14/check_point/stage4_snapshot_basebot_0.506462.pth'
    # gappl.do_prediction(target_path)

    G.logger.info('success!')
