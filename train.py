import os
import time
import yaml
import torch
import argparse
import pandas as pd
import numpy as np
import torch.utils.data
import torch.backends.cudnn as cudnn
from time import time as t
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from models import alexnet, vgg, resnet, googlenet, se_resnet
from utils.utils import ObjectDict, AverageMeter, save_checkpoint
from utils.dataset import get_dataset
from utils.metrics import f1_score, map_score


parser = argparse.ArgumentParser(description='Pytorch CelebA Implementation')
parser.add_argument('--cfg', default='', type=str, metavar='PATH',
                    help='path to configuration (default: none)')
parser.add_argument('--gpu', default=0, type=int,
                    help='the id of gpu to train')

def train_one_epoch(model, criterion, optimizer, epoch, train_dataloader, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f1_micros = AverageMeter()
    f1_macros = AverageMeter()
    maps = AverageMeter()

    model.train()

    end = t()

    for batch_idx, (input, target) in enumerate(train_dataloader):
        # process data loading time
        data_time.update(t() - end)

        # model execution
        input = input.to(device)
        target = target.to(device).float().requires_grad_()
        output = model(input)

        loss = criterion(output, target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # process output and other measurements
        loss = loss.float()
        output = torch.sigmoid(output)
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        cur_f1_micro, cur_f1_macro = f1_score(y_pred=output, y_true=target)
        cur_map = map_score(y_pred=output, y_true=target)
        losses.update(loss, input.size(0))
        f1_micros.update(cur_f1_micro, input.size(0))
        f1_macros.update(cur_f1_macro, input.size(0))
        maps.update(cur_map, input.size(0))

        # process batch time
        batch_time.update(t() - end)
        end = t()

        if batch_idx % cfg.print_frequency == 0:
            print(f'Epoch: [{epoch}]/[{cfg.epochs}],\t'
                  f'Batch: [{batch_idx}]/[{len(train_dataloader)}],\t'
                  f'Time: {batch_time.val:.4f} ({batch_time.avg:.4f}),\t'
                #   f'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}),\t'
                  f'map: {maps.val:.4f} ({maps.avg:.4f})\t'
                  f'f1_score_micro: {f1_micros.val:.4f} ({f1_micros.avg:.4f})\t'
                  f'f1_score_macro: {f1_macros.val:.4f} ({f1_macros.avg:.4f})')

    return losses.avg, maps.avg, f1_micros.avg, f1_macros.avg

@torch.no_grad()
def validate(model, criterion, epoch, val_dataloader, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f1_micros = AverageMeter()
    f1_macros = AverageMeter()
    maps = AverageMeter()

    model.eval()

    end = t()

    for batch_idx, (input, target) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f' * Validating (epoch: {epoch})'):
        # process data loading time
        data_time.update(t() - end)

        # model execution
        input = input.to(device)
        target = target.to(device).float()
        output = model(input)
        loss = criterion(output, target)

        # process output and other measurements
        loss = loss.float()
        output = torch.sigmoid(output)
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        cur_f1_micro, cur_f1_macro = f1_score(y_pred=output, y_true=target)
        cur_map = map_score(y_pred=output, y_true=target)
        losses.update(loss, input.size(0))
        f1_micros.update(cur_f1_micro, input.size(0))
        f1_macros.update(cur_f1_macro, input.size(0))
        maps.update(cur_map, input.size(0))

        # process batch time
        batch_time.update(t() - end)
        end = t()

        # print(f'Epoch: [{epoch}]/[{cfg.epochs}]\t'
        #       f'Batch: [{batch_idx}]/[{len(val_dataloader)}]\t'
        #       f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        #     #   f'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
        #       f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
        #       f'f1_score {f1_micro.val:.4f} {f1_macro.val:.4f}')

    print(f'* Epoch: {epoch}, map: {maps.avg:.4f}, f1_score: {f1_micros.avg:.4f}, {f1_macros.avg:.4f}, val_loss: {losses.avg:.4f}\n')

    return losses.avg, maps.avg, f1_micros.avg, f1_macros.avg

def train(model, criterion, optimizer, lr_scheduler, train_dataloader, val_dataloader, writer, cfg):
    best_f1_macro = 0
    best_f1_micro = 0
    best_map = 0
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(cfg.epochs):
        train_loss, train_map, train_f1_micro, train_f1_macro = train_one_epoch(model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, train_dataloader=train_dataloader, cfg=cfg)
        val_loss, val_map, val_f1_micro, val_f1_macro = validate(model=model, criterion=criterion, epoch=epoch, val_dataloader=val_dataloader, cfg=cfg)

        lr_scheduler.step()

        if cfg.multigpu:
            save_dict = {
                'epoch': epoch + 1,
                'loss': val_loss,
                'score': (val_f1_micro, val_f1_macro),
                'state_dict': model.module.state_dict()
            }
        else:
            save_dict = {
                'epoch': epoch + 1,
                'loss': val_loss,
                'score': (val_f1_micro, val_f1_macro),
                'state_dict': model.state_dict()
            }

        is_best = False
        if val_f1_micro > best_f1_micro:
            best_f1_macro = val_f1_macro
            best_f1_micro = val_f1_micro
            best_val_loss = val_loss
            best_map = val_map
            best_epoch = epoch
            is_best = True
        save_checkpoint(epoch, save_dict, is_best, cfg, path=cfg.checkpoint_path)

        writer.add_scalar(f'loss/train_loss', train_loss, epoch)
        writer.add_scalar(f'loss/val_loss', val_loss, epoch)
        writer.add_scalar(f'f1_micro/train_f1_micro', train_f1_micro, epoch)
        writer.add_scalar(f'f1_micro/val_f1_micro', val_f1_micro, epoch)
        writer.add_scalar(f'f1_micro/train_f1_macro', train_f1_macro, epoch)
        writer.add_scalar(f'f1_macro/val_f1_macro', val_f1_macro, epoch)
        writer.add_scalar(f'map/train_map', train_map, epoch)
        writer.add_scalar(f'map/val_map', val_map, epoch)

    print(f"** best val loss: {best_val_loss:.4f}, best f1 micro: {best_f1_micro:.4f}, best f1 macro: {best_f1_macro:.4f}, best map: {best_map:.4f}, best epoch: {best_epoch}")

if __name__ == '__main__':
    # load arguments and configurations, and then convert them to object
    args = parser.parse_args()
    with open(args.cfg, 'r') as configure_file:
        cfg_dict = yaml.load(configure_file, Loader=yaml.FullLoader)
    cfg = ObjectDict(cfg_dict)
    print(cfg)

    # make some preparations
    pid = f"{cfg.network}{cfg.get('layers', '')}_{cfg.learning_rate}_{time.strftime('%d_%H_%M_%S', time.localtime())}"
    cfg['checkpoint_path'] = f'./states/{pid}'
    if not os.path.exists(cfg['checkpoint_path']):
        os.makedirs(cfg['checkpoint_path'])
    # cudnn.benchmark = True
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # define summary write
    writer = SummaryWriter(f'./docs/tensorboard_logs/{pid}')

    # load datasets
    train_dataset, val_dataset, _ = get_dataset(name='CelebA', path=cfg.data_dir)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                   num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers, pin_memory=True)
    # load model
    if cfg.network == 'Alexnet':
        model = alexnet(num_classes=40)
    elif cfg.network == 'VGG':
        model = vgg(layers=cfg.layers, num_classes=40)
    elif cfg.network == 'ResNet':
        model = resnet(layers=cfg.layers, num_classes=40)
    elif cfg.network == 'GoogLeNet':
        model = googlenet(num_classes=40)
    elif cfg.network == 'SEResNet':
        model = se_resnet(layers=cfg.layers, num_classes=40)
    else:
        raise NotImplementedError

    model = model.to(device)

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
    # define criterion
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    train(model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, train_dataloader=train_dataloader, val_dataloader=val_dataloader, writer=writer, cfg=cfg)