#!/usr/bin/env python

from collections import OrderedDict
import argparse
import importlib
import json
import logging
import pathlib
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torchvision
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import get_loader
from config import get_default_config

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.train.device = 'cpu'
    config.freeze()
    return config


def load_model(config):
    module = importlib.import_module(config.model.name)
    Network = getattr(module, 'Network')
    return Network(config)


def get_optimizer(config, model):
    if config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.train.base_lr,
            momentum=config.train.momentum,
            weight_decay=config.train.weight_decay,
            nesterov=config.train.nesterov)
    else:
        raise ValueError()
    return optimizer


def get_scheduler(config, optimizer):
    if config.scheduler.name == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.scheduler.multistep.milestones,
            gamma=config.scheduler.multistep.lr_decay)
    elif config.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.train.epochs, config.scheduler.cosine.lr_min)
    else:
        raise ValueError()
    return scheduler


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def train(epoch, model, optimizer, scheduler, criterion, train_loader, config,
          writer):
    global global_step

    logger.info(f'Train {epoch}')

    model.train()
    device = torch.device(config.run.device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        if config.run.tensorboard and step == 0:
            image = torchvision.utils.make_grid(
                data, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        acc_meter.update(accuracy, num)

        if config.run.tensorboard:
            writer.add_scalar('Train/RunningLoss', loss_, global_step)
            writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)

        if step % 100 == 0:
            logger.info(f'Epoch {epoch} Step {step}/{len(train_loader)} '
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'Accuracy {acc_meter.val:.4f} ({acc_meter.avg:.4f})')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    if config.run.tensorboard:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Accuracy', acc_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)
        writer.add_scalar('Train/lr', scheduler.get_lr()[0], epoch)

    train_log = OrderedDict({
        'epoch':
        epoch,
        'train':
        OrderedDict({
            'loss': loss_meter.avg,
            'accuracy': acc_meter.avg,
            'time': elapsed,
        }),
    })
    return train_log


def test(epoch, model, criterion, test_loader, config, writer):
    logger.info(f'Test {epoch}')

    model.eval()
    device = torch.device(config.run.device)

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(test_loader):
            if config.run.tensorboard and epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(
                    data, normalize=True, scale_each=True)
                writer.add_image('Test/Image', image, epoch)

            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, dim=1)

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)

    logger.info(
        f'Epoch {epoch} Loss {loss_meter.avg:.4f} Accuracy {accuracy:.4f}')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    if config.run.tensorboard:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    test_log = OrderedDict({
        'epoch':
        epoch,
        'test':
        OrderedDict({
            'loss': loss_meter.avg,
            'accuracy': accuracy,
            'time': elapsed,
        }),
    })
    return test_log


def main():
    # parse command line arguments
    config = load_config()
    logger.info(json.dumps(config, indent=2))

    # set random seed
    seed = config.run.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # create output directory
    outdir = pathlib.Path(config.run.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # TensorBoard SummaryWriter
    writer = SummaryWriter(
        outdir.as_posix()) if config.run.tensorboard else None

    # save config
    with open(outdir / 'config.yaml', 'w') as fout:
        fout.write(str(config))

    # data loaders
    train_loader, test_loader = get_loader(config)

    # model
    model = load_model(config)
    model.to(torch.device(config.run.device))
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info(f'n_params: {n_params}')

    criterion = nn.CrossEntropyLoss(reduction='mean')

    # optimizer
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    # run test before start training
    test(0, model, criterion, test_loader, config, writer)

    epoch_logs = []
    for epoch in range(config.train.epochs):
        epoch += 1
        scheduler.step()

        train_log = train(epoch, model, optimizer, scheduler, criterion,
                          train_loader, config, writer)
        test_log = test(epoch, model, criterion, test_loader, config, writer)

        epoch_log = train_log.copy()
        epoch_log.update(test_log)
        epoch_logs.append(epoch_log)
        with open(outdir / 'log.json', 'w') as fout:
            json.dump(epoch_logs, fout, indent=2)

        state = OrderedDict([
            ('config', config),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('accuracy', test_log['test']['accuracy']),
        ])
        model_path = outdir / 'model_state.pth'
        torch.save(state, model_path)


if __name__ == '__main__':
    main()
