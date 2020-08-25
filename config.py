import argparse
import numpy as np
import os
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

class Config(object):
    data_root = '/DATA7_DB7/data/sxfeng/Program/Deep-S'
    model_root = '/DATA7_DB7/data/sxfeng/Checkpoints/Deep-S/checkpoints'

    train_paths = os.path.join(data_root, 'Dataset/collapse_csv/trainVB.csv')
    val_paths = os.path.join(data_root, 'Dataset/collapse_csv/valVB.csv')
    test_paths = os.path.join(data_root, 'Dataset/collapse_csv/testVB.csv')

    save_model_dir = 'resnet18'
    save_model_name = 'resnet18_1.pth'
    resume = False  # resume from checkpoint
    # net = None   # checkpoint path of network
    net = os.path.join(model_root, 'resnet18/resnet18_1/ckpt_epoch22.pth')

    n_classes = 3
    lr = 0.0001
    model = 'resnet32'
    batch_size = 64
    epoch = 100
    start_save = 20
    seed = None
    weight_decay = 1e-5
    nGPUs = 1

    padding = True

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn(f'Warning: config has no attribute {k}')
            setattr(self, k, v)

ARGS = Config()

if ARGS.seed is not None:
    SEED = ARGS.seed
else:
    SEED = np.random.randint(10000)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

N_GPUS = ARGS.nGPUs
N_CLASSES = ARGS.n_classes
BATCH_SIZE = ARGS.batch_size

LR = ARGS.lr
EPOCH = ARGS.epoch

def adjust_learning_rate(optimizer, lr_init, epoch):
    """decrease the learning rate at 160 and 180 epoch ( from LDAM-DRW, NeurIPS19 )"""
    lr = lr_init

    if epoch < 5:
        lr = (epoch + 1) * lr_init / 5
    else:
        if epoch >= 160:
            lr /= 100
        if epoch >= 180:
            lr /= 100

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
