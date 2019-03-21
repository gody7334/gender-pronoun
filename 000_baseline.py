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

mode="DEV"

##################################################

import logging
from pathlib import Path
import colored_traceback.always
from matplotlib.pyplot import *

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from utils.bot import OneCycle
from utils.lr_finder import LRFinder
from utils.lr_scheduler import TriangularLR

from dev.gap_bot import *
from dev.data import *
from dev.model import *

####################################################

G.logger.info("load model")
model = GAPModel(BERT_MODEL, torch.device("cuda:0"))

G.logger.info("load gapdl")
gapdl = GAPDataLoader()

G.logger.info("create bot")
bot = GAPBot(
    model, gapdl.train_loader, gapdl.val_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3),
    criterion=torch.nn.CrossEntropyLoss(),
    echo=True,
    use_tensorboard=True,
    avg_window=25,
    snapshot_policy='last',
)

'stage0'
oc = OneCycle(bot,
        scheduler="Default Triangular",
        unfreeze_layers=[model.head],
        n_epoch=5,
        stage='0',
        )
oc.train_one_cycle()

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

