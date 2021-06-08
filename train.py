import torch
import yaml
import argparse
from utils import ObjectDict, AverageMeter, save_checkpoint
import torch.utils.data
from utils.dataset import get_dataset
import torch.backends.cudnn as cudnn
import models
import numpy as np
from time import time as t
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='Pytorch CelebA Implementation')
parser.add_argument('--cfg', default='', type=str, metavar='PATH',
                    help='path to configuration (default: none)')

args = parser.parse_args()

# load configurations and convert it to object
with open(args.cfg, 'r') as configure_file:
    cfg_dict = yaml.load(configure_file, Loader=yaml.FullLoader)
cfg = ObjectDict(cfg_dict)

# model, optimizer and lr_scheduler
cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet50(pretrained=False, num_classes=cfg.num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), betas=(cfg.adam_beta1, cfg.adam_beta2), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma, last_epoch=cfg.epochs)

# define criterion
criterion = torch.nn.BCEWithLogitsLoss()

# dataset
train_dataset, val_dataset, _ = get_dataset(name='CelebA', path=cfg.data_dir)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False,
                                             num_workers=cfg.num_worker, pin_memory=True)

def train_one_epoch(epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f1_micro = AverageMeter()
    f1_macro = AverageMeter()

    model.train()

    end = t()

    for batch_idx, (input, target) in enumerate(train_dataloader):
        # process data loading time
        data_time.update(t() - end)

        # model execution
        input = input.to(device)
        output = model(input)
        loss = criterion(output, target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # process output and other measurements
        loss = loss.float()
        output = output.detach().cpu().numpy()
        target = target.numpy()
        cur_f1_micro, cur_f1_macro = score(y_pred=output, y_true=target, rounded=False)
        losses.update(loss, input.size(0))
        f1_micro.update(cur_f1_micro, input.size(0))
        f1_macro.update(cur_f1_macro, input.size(0))

        # process batch time
        batch_time.update(t() - end)
        end = t()

        print(f'Epoch: [{epoch}]/[{cfg.epochs}]\t'
              f'Batch: [{batch_idx}]/[{len(train_dataloader)}]\t'
              f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
              f'Loss {loss.val:.3f}'
              f'f1_score {f1_micro.val:.3f} {f1_macro.val:.3f}')

    lr_scheduler.step()

    return losses.avg, f1_micro.avg, f1_macro.avg

def validate(epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f1_micro = AverageMeter()
    f1_macro = AverageMeter()

    model.eval()

    end = t()

    for batch_idx, (input, target) in enumerate(val_dataloader):
        # process data loading time
        data_time.update(t() - end)

        # model execution
        input = input.to(device)
        output = model(input)
        loss = criterion(output, target)

        # process output and other measurements
        loss = loss.float()
        output = output.detach().cpu().numpy()
        target = target.numpy()
        cur_f1_micro, cur_f1_macro = score(y_pred=output, y_true=target, rounded=False)
        losses.update(loss, input.size(0))
        f1_micro.update(cur_f1_micro, input.size(0))
        f1_macro.update(cur_f1_macro, input.size(0))

        # process batch time
        batch_time.update(t() - end)
        end = t()

        print(f'Epoch: [{epoch}]/[{cfg.epochs}]\t'
              f'Batch: [{batch_idx}]/[{len(train_dataloader)}]\t'
              f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
              f'Loss {loss.val:.3f}'
              f'f1_score {f1_micro.val:.3f} {f1_macro.val:.3f}')

    print(f'* Epoch: {epoch}, f1_score: {f1_micro.val:.3f} {f1_macro.val:.3f}')

    return losses.avg, f1_micro.avg, f1_macro.avg

def train():
    best_f1_macro = 0
    for epoch in range(cfg.epochs):
        train_loss, train_f1_micro, train_f1_macro = train_one_epoch(epoch)
        val_loss, val_f1_micro, val_f1_macro = validate(epoch)

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
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            is_best = True
        save_checkpoint(save_dict, is_best, cfg)


def score(y_pred, y_true, rounded=False):
    if not rounded:
        y_pred = np.round(y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    return f1_micro, f1_macro

if __name__ == '__main__':
    train()