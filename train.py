import os
import socket
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import models
import quan_ops
from misc.losses import CrossEntropyLossSoft, KurtosisLoss
from datasets.data import get_dataset, get_transform
from misc.optimizer import get_optimizer_config, get_lr_scheduler
from misc.utils import setup_logging, setup_gpus
from misc.utils import results_dir_config, check_resume_pretrain, freeze_param, save_ckpt
from misc.utils import AverageMeter, accuracy
from torchsummary import summary

from configs.config import *

# Put in the MIG UUID to use the MIG instance
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-b978da18-95fe-5992-93bb-7abcb371f386"
args = all_cfg

def main():
    global args
    if wandb_cfg.wandb_enabled:
        import wandb
        run = wandb.init()
        args = wandb.config
    else:
        args = all_cfg

    quan_ops.conv2d_quan_ops.args = args
    models.resnet_quan.args = args

    weight_bit_width = list(map(int, args.weight_bit_width.split(',')))
    act_bit_width = list(map(int, args.act_bit_width.split(',')))
    cal_bw = list(map(int, args.cal_bit_width.split(',')))
    lr_decay = list(map(int, args.lr_decay.split(',')))

    results_dir = results_dir_config(args.log_id, args.model)
    hostname = socket.gethostname()
    setup_logging(os.path.join(results_dir, 'log_{}.txt'.format(hostname)))
    logging.info("running arguments: %s", args)

    # best_gpu = setup_gpus()
    best_gpu = 0
    # torch.cuda.set_device(best_gpu)
    torch.backends.cudnn.benchmark = True

    train_transform = get_transform(args.dataset, 'train')
    train_data = get_dataset(args.dataset, args.train_split, train_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_transform = get_transform(args.dataset, 'val')
    val_data = get_dataset(args.dataset, 'val', val_transform)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    model = models.__dict__[args.model](wbit_list=weight_bit_width, 
                                        abit_list=act_bit_width,
                                        num_classes=train_data.num_classes).cuda()

    # summary(model, (3, 32, 32))
    # assert 1==2

    print(model)
    optimizer = get_optimizer_config(model, args.optimizer, args.lr, args.weight_decay)
    best_prec1, lr_scheduler, start_epoch = check_resume_pretrain(model, optimizer, best_gpu, results_dir)
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_soft = CrossEntropyLossSoft().cuda()

    if lr_scheduler is None:
        lr_scheduler = get_lr_scheduler(args.scheduler, optimizer, lr_decay)
    
    freeze_param(model, cal_bw)     # freeze parameter
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    num_epochs = 1 if args.is_training == 'F' else args.epochs

    for epoch in range(start_epoch, num_epochs):
        if args.is_training == 'T':
            model.train()
            train_loss, train_prec1, train_prec5 = forward(train_loader, model, criterion, criterion_soft, epoch, True, optimizer)
            train_loss_dict, train_prec1_dict, train_prec5_dict = [{bw: loss for bw, loss in zip(weight_bit_width, values)} for values in [train_loss, train_prec1, train_prec5]]
        model.eval()
        val_loss, val_prec1, val_prec5 = forward(val_loader, model, criterion, criterion_soft, epoch, False)
        val_loss_dict, val_prec1_dict, val_prec5_dict = [{bw: loss for bw, loss in zip(weight_bit_width, values)} for values in [val_loss, val_prec1, val_prec5]]

        if args.is_training == 'T':
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()

            is_best = val_prec1[-1] > best_prec1 if best_prec1 is not None else True
            best_prec1 = max(val_prec1[-1], best_prec1) if best_prec1 is not None else val_prec1[-1]
            save_ckpt(epoch, model, best_prec1, optimizer, is_best, path=results_dir + '/ckpt')
            
            max_bw = -1 if args.is_calibrate == "F" else cal_bw[-1] - 1
            tqdm.write('Epoch {}: \ntrain loss {:.2f}, train prec1 {:.2f}, train prec5 {:.2f}\n'
                        '  val loss {:.2f},   val prec1 {:.2f},   val prec5 {:.2f}'.format(epoch, 
                        train_loss[max_bw], train_prec1[max_bw], train_prec5[max_bw], 
                        val_loss[max_bw], val_prec1[max_bw], val_prec5[max_bw]))
            if wandb_cfg.wandb_enabled:
                wandb_ez.log({"num_param": num_parameters,
                            "curr_lr": lr_scheduler.get_last_lr()[0],
                            "train_loss": train_loss_dict, 
                            "train_prec1": train_prec1_dict,
                            "train_prec5": train_prec5_dict,
                            "val_loss": val_loss_dict,
                            "val_prec1": val_prec1_dict,
                            "val_prec5": val_prec5_dict})
                wandb_ez.upload_model(results_dir + '/ckpt')

        else: 
            if wandb_cfg.wandb_enabled:
                wandb_ez.log({"num_param": num_parameters,
                            "curr_lr": lr_scheduler.get_last_lr()[0],
                            "val_loss": val_loss_dict,
                            "val_prec1": val_prec1_dict,
                            "val_prec5": val_prec5_dict})

        for w_bw, a_bw, vl, vp1, vp5 in zip(weight_bit_width, act_bit_width, val_loss, val_prec1, val_prec5):
            tqdm.write('wbit {}, abit {}: val loss {:.2f},   val prec1 {:.2f},   val prec5 {:.2f}'.format(w_bw, a_bw, vl, vp1, vp5))

def forward(data_loader, model, criterion, criterion_soft, epoch, training=True, optimizer=None):
    weight_bit_width = list(map(int, args.weight_bit_width.split(',')))
    act_bit_width = list(map(int, args.act_bit_width.split(',')))
    cal_bw = list(map(int, args.cal_bit_width.split(',')))

    losses = [AverageMeter() for _ in weight_bit_width]
    top1 = [AverageMeter() for _ in weight_bit_width]
    top5 = [AverageMeter() for _ in weight_bit_width]

    for i, (input, target) in enumerate(data_loader):
        if not training:
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda(non_blocking=True)

                for w_bw, a_bw, am_l, am_t1, am_t5 in zip(weight_bit_width, act_bit_width, losses, top1, top5):
                    model.apply(lambda m: setattr(m, 'wbit', w_bw))
                    model.apply(lambda m: setattr(m, 'abit', a_bw))
                    output = model(input)
                    loss = criterion(output, target)

                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                    am_l.update(loss.item(), input.size(0))
                    am_t1.update(prec1.item(), input.size(0))
                    am_t5.update(prec5.item(), input.size(0))
        else:
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad()

            if args.is_calibrate == "F":
                # train full-precision supervisor
                model.apply(lambda m: setattr(m, 'wbit', weight_bit_width[-1]))
                model.apply(lambda m: setattr(m, 'abit', act_bit_width[-1]))

                # loss = 0.015 * KurtosisLoss(model)
                # loss.backward()

                output = model(input)
                #print("output.shape:", output.shape)
                loss = criterion(output, target)
                
                loss.backward()
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses[-1].update(loss.item(), input.size(0))
                top1[-1].update(prec1.item(), input.size(0))
                top5[-1].update(prec5.item(), input.size(0))

                target_soft = torch.nn.functional.softmax(output.detach(), dim=1)

            # train less-bit-wdith models
            for w_bw, a_bw, am_l, am_t1, am_t5 in zip(weight_bit_width[:-1][::-1], act_bit_width[:-1][::-1], losses[:-1][::-1], top1[:-1][::-1], top5[:-1][::-1]):

                if (args.is_calibrate == "T") and (w_bw not in cal_bw):
                    continue

                model.apply(lambda m: setattr(m, 'wbit', w_bw))
                model.apply(lambda m: setattr(m, 'abit', a_bw))

                output = model(input)
                if args.is_calibrate == "T":
                    loss = criterion(output, target)
                else:
                    loss = criterion_soft(output, target_soft)
                loss.backward()
                target_soft = torch.nn.functional.softmax(output.detach(), dim=1)

                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                am_l.update(loss.item(), input.size(0))
                am_t1.update(prec1.item(), input.size(0))
                am_t5.update(prec5.item(), input.size(0))
            optimizer.step()

            if i % args.print_freq == 0:
                max_bw = -1 if args.is_calibrate == "F" else cal_bw[-1] - 1
                tqdm.write('epoch {0}, iter {1}/{2}, bit_width_max loss {3:.2f}, prec1 {4:.2f}, prec5 {5:.2f}'.format(
                    epoch, i, len(data_loader), losses[max_bw].val, top1[max_bw].val, top5[max_bw].val))

    return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5]

if __name__ == '__main__':
    if wandb_cfg.wandb_enabled:
        from wandb_ez import wandb_ez
        # from wandb_ez.wandb_cfg import *
        run = wandb_ez.init(args, main)
        # if wandb_cfg['sweep'] is False:
            # main()
    else:
        main()