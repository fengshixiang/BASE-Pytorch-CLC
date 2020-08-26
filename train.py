import csv
import os
import numpy as np
from tqdm import tqdm
import fire

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchnet import meter


from utils import Logger, progress_bar
from config import *
from models import UNet, ResNet18
from dataloader import My_Dataset


def save_checkpoint(acc, model, optim, epoch, index=False):
    # Save checkpoint.
    print('Saving..')

    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        'net': model.state_dict(),
        'optimizer': optim.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    if index:
        ckpt_name = 'ckpt_epoch' + str(epoch) + '.pth'
    else:
        ckpt_name = 'ckpt_' + '.pth'

    ckpt_path = os.path.join(ARGS.model_root, ARGS.save_model_dir, ARGS.save_model_name.split('.')[0], ckpt_name)
    torch.save(state, ckpt_path)


def train_epoch(net, criterion, optimizer, data_loader, logger=None):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets, img_path) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        score = net(inputs)
        loss = criterion(score, targets).mean()

        train_loss += loss.item() * batch_size
        predicted = score.max(1)[1]
        total += batch_size
        correct += predicted.eq(targets).float().sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar(i, len(data_loader), 'train')

    msg = '  Train loss: %.3f| Acc: %.3f%% (%d/%d)' % \
          (train_loss / total, 100. * correct / total, correct, total)
    if logger:
        logger.log(msg)
    else:
        print(msg)

    return train_loss / total, 100. * correct / total


def evaluate(net, dataloader, logger=None):
    is_training = net.training
    net.eval()
    criterion = nn.CrossEntropyLoss()

    val_cm = meter.ConfusionMeter(N_CLASSES)
    val_mAP = meter.mAPMeter()

    total_loss = 0.0
    total = 0.0

    with torch.no_grad():
        for inputs, targets, img_path in dataloader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            score = net(inputs)
            loss = criterion(score, targets)

            total_loss += loss.item() * batch_size
            total += batch_size

            # *********************** confusion matrix and mAP ***********************
            one_hot = torch.zeros(targets.size(0), 3).scatter_(1, targets.data.cpu().unsqueeze(1), 1)
            val_cm.add(F.softmax(score, dim=1).data, targets.data)
            val_mAP.add(F.softmax(score, dim=1).data, one_hot)

    val_cm = val_cm.value()
    val_acc = 100. * sum([val_cm[c][c] for c in range(N_CLASSES)]) / val_cm.sum()
    val_sp = [100. * (val_cm.sum() - val_cm.sum(0)[i] - val_cm.sum(1)[i] + val_cm[i][i]) / (val_cm.sum() - val_cm.sum(1)[i])
                for i in range(N_CLASSES)]
    val_se = [100. * val_cm[i][i] / val_cm.sum(1)[i] for i in range(N_CLASSES)]


    results = {
        'val_loss': total_loss / total,
        'acc': val_acc
    }

    msg = '  Val   loss: %.3f | Acc: %.3f%% (%d)' % \
          (results['val_loss'], results['acc'], total)
    if logger:
        logger.log(msg)
    else:
        print(msg)

    net.train(is_training)
    return results


def train():
    model_save_path = os.path.join(ARGS.model_root, ARGS.save_model_dir,
                                   ARGS.save_model_name.split('.')[0])
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    logger = Logger(model_save_path)
    LOGDIR = logger.logdir

    # log args
    args_msg = 'ARGS \r'\
               'train_path: {} \r'\
               'val_path:   {} \r'\
               'N_CLASSES: {}, LR: {}, MODEL: {}, EPOCH: {}, START_SAVE: {} \r'\
               'SEED: {}, WEIGHT_DECAY: {}, NGPUS: {}'.format(ARGS.train_paths, ARGS.val_paths,
               N_CLASSES, LR, ARGS.model, EPOCH, ARGS.start_save, SEED, ARGS.weight_decay, ARGS.nGPUs)
    logger.log(args_msg)

    LOG_CSV = os.path.join(model_save_path, 'log_{}.csv'.format(SEED))
    LOG_CSV_HEADER = [
        'epoch', 'train loss', 'train acc', 'val loss', 'val acc'
    ]
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(LOG_CSV_HEADER)

    BEST_VAL = 0  # best validation accuracy
    BEST_EPOCH = 0

    logger.log('==> Building model: %s' % ARGS.save_model_dir)

    train_data = My_Dataset(ARGS.train_paths, phase='train', num_classes=N_CLASSES, padding=ARGS.padding)
    val_data = My_Dataset(ARGS.test_paths, phase='test', num_classes=N_CLASSES, padding=ARGS.padding)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    net = ResNet18(N_CLASSES)
    net = net.to(device)
    per_cls_weights = torch.ones(N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=ARGS.weight_decay)

    START_EPOCH = 0
    if ARGS.resume:
        # Load checkpoint.
        logger.log('==> Resuming from checkpoint..')

        if ARGS.net is not None:
            ckpt_t = torch.load(ARGS.net)
            net.load_state_dict(ckpt_t['net'])
            optimizer.load_state_dict(ckpt_t['optimizer'])
            START_EPOCH = ckpt_t['epoch'] + 1

    if N_GPUS > 1:
        logger.log('Multi-GPU mode: using %d GPUs for training.' % N_GPUS)
        net = nn.DataParallel(net)
    elif N_GPUS == 1:
        logger.log('Single-GPU mode.')

    for epoch in range(START_EPOCH, EPOCH):
        logger.log('** Epoch %d: %s' % (epoch, LOGDIR))

        adjust_learning_rate(optimizer, LR, epoch)

        train_loss, train_acc = train_epoch(net, criterion, optimizer, train_loader, logger)
        train_stats = {'train_loss': train_loss, 'train_acc': train_acc}

        ## Evaluation ##
        train_eval = evaluate(net, train_loader, logger=logger)
        val_eval = evaluate(net, val_loader, logger=logger)
        val_acc = val_eval['val_acc']
        if epoch >= ARGS.start_save:
            if val_acc >= BEST_VAL:
                BEST_VAL = val_acc
                BEST_EPOCH = epoch
                save_checkpoint(val_acc, net, optimizer, epoch, True)
                logger.log("  Val performance acc : {}".format(val_acc))

        def _convert_scala(x):
            if hasattr(x, 'item'):
                x = x.item()
            return x

        log_tr = ['train_loss', 'train_acc']
        log_val = ['val_loss', 'val_acc']

        log_vector = [epoch] + [train_stats.get(k, 0) for k in log_tr] + [val_eval.get(k, 0) for k in log_val]
        log_vector = list(map(_convert_scala, log_vector))

        with open(LOG_CSV, 'a') as f:
            logwriter = csv.writer(f, delimiter=',')
            logwriter.writerow(log_vector)

    logger.log(' * %s' % LOGDIR)
    logger.log("Best Accuracy : {}".format(BEST_VAL))


def test():
    logdir = ARGS.net.split(ARGS.net.split('/')[-1])[0]
    logger = Logger(logdir, train=False)
    LOGDIR = logger.logdir

    test_data = My_Dataset(ARGS.test_paths, phase='test', num_classes=N_CLASSES, padding=ARGS.padding)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    net = ResNet18(N_CLASSES)
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()

    ckpt_t = torch.load(ARGS.net)
    net.load_state_dict(ckpt_t['net'])
    BEST_EPOCH = ckpt_t['epoch']

    # log args
    args_msg = 'ARGS \rBEST_EPOCH : {} \r'\
               'test_path: {} \r'\
               'MODEL_LOAD_PATH: {}\r '.format(BEST_EPOCH, ARGS.test_paths, ARGS.net)
    logger.log(args_msg)

    test_cm = meter.ConfusionMeter(N_CLASSES)
    test_mAP = meter.mAPMeter()

    total_loss = 0.0
    total = 0.0

    net.eval()

    with torch.no_grad():
        for inputs, targets, img_path in test_loader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            score = net(inputs)
            loss = criterion(score, targets)

            total_loss += loss.item() * batch_size
            total += batch_size

            # *********************** confusion matrix and mAP ***********************
            one_hot = torch.zeros(targets.size(0), 3).scatter_(1, targets.data.cpu().unsqueeze(1), 1)
            test_cm.add(F.softmax(score, dim=1).data, targets.data)
            test_mAP.add(F.softmax(score, dim=1).data, one_hot)

    test_cm = test_cm.value()
    test_acc = 100. * sum([test_cm[c][c] for c in range(N_CLASSES)]) / test_cm.sum()
    test_sp = [100. * (test_cm.sum() - test_cm.sum(0)[i] - test_cm.sum(1)[i] + test_cm[i][i]) / (test_cm.sum() - test_cm.sum(1)[i])
                for i in range(N_CLASSES)]
    test_se = [100. * test_cm[i][i] / test_cm.sum(1)[i] for i in range(N_CLASSES)]


    results = {
        'loss': total_loss / total,
        'acc': test_acc
    }

    msg = 'Test   loss: %.3f | Acc: %.3f%% (%d)' % \
          (results['loss'], results['acc'], total)
    if logger:
        logger.log(msg)
    else:
        print(msg)

    logger.log("Best Accuracy : {}".format(test_acc))


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
    })
