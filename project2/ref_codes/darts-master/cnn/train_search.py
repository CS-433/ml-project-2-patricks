import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

import datasets as DS
from learning_rate_scheduling import LearningRateScheduler

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset_path', type=str, default='/storage/data/classification_dataset_balanced/', help='location of the data corpus')
parser.add_argument('--class_num', type=int, default=6, help='dataset class number')
parser.add_argument('--patch_size', type=int, default=224, help='patch size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--workers', type=int, default=8, help='workers')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--lr-wr-epochs', default=10, type=int, help='length of first warm restart cycle (default: 10)')
parser.add_argument('--lr-wr-mul', default=2, type=int, help='scaling factor for warm restarts (default: 2)')
parser.add_argument('--lr-wr-min', default=1e-5, type=float, help='minimum learning rate (default: 1e-5)')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.BCELoss(reduction='mean')
  criterion = criterion.cuda()
  model = Network(args.init_channels, args.class_num, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=args.learning_rate,
    momentum=args.momentum, 
    weight_decay=args.weight_decay)

  dataset = DS.CODEBRIM(torch.cuda.is_available(), args)

  train_queue = dataset.train_loader

  valid_queue = dataset.val_loader

  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=args.lr_wr_epochs,
   T_mult=args.lr_wr_mul, eta_min=args.lr_wr_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    hard_prec, soft_prec = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train multi-target(hard) accuracy %f', hard_prec.avg)
    logging.info('train individual(soft) accuracy %f', soft_prec.avg)

    # validation
    hard_prec, soft_prec = infer(valid_queue, model, criterion)
    logging.info('valid multi-target(hard) accuracy %f', hard_prec.avg)
    logging.info('valid individual(soft) accuracy %f', soft_prec.avg)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  losses = utils.AverageMeter()
  hard_prec = utils.AverageMeter()
  soft_prec = utils.AverageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    output = torch.sigmoid(logits)
    # scale the loss by the ratio of the split batch size and the original
    loss = criterion(output, target) * input.size(0) / float(args.batch_size)

    # update the 'losses' meter with the actual measure of the loss
    losses.update(loss.item() * args.batch_size / float(input.size(0)), input.size(0))
    # compute performance measures
    output = output >= 0.5  # binarizing sigmoid output by thresholding with 0.5
    equality_matrix = (output.float() == target).float()
    hard = torch.mean(torch.prod(equality_matrix, dim=1)) * 100.
    soft = torch.mean(equality_matrix) * 100.
    # update peformance meters
    hard_prec.update(hard.item(), input.size(0))
    soft_prec.update(soft.item(), input.size(0))

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    if step % args.report_freq == 0:
      logging.info(f'train step: {step:03d}\t'+
      f'hard prec {hard_prec.val:.3f} ({hard_prec.avg:.3f})\t'+
      f'soft prec {soft_prec.val:.3f} ({soft_prec.avg:.3f})\t')

  return hard_prec, soft_prec


def infer(valid_queue, model, criterion):
  losses = utils.AverageMeter()
  hard_prec = utils.AverageMeter()
  soft_prec = utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(non_blocking=True)

    logits = model(input)
    loss = criterion(logits, target)
    output_probs = torch.sigmoid(logits)
    # scale the loss by the ratio of the split batch size and the original
    loss = criterion(output_probs, target) * input.size(0) / float(args.batch_size)

    # update the 'losses' meter with the actual measure of the loss
    losses.update(loss.item() * args.batch_size / float(input.size(0)), input.size(0))
    # compute performance measures
    output = output >= 0.5  # binarizing sigmoid output by thresholding with 0.5
    equality_matrix = (output.float() == target).float()
    hard = torch.mean(torch.prod(equality_matrix, dim=1)) * 100.
    soft = torch.mean(equality_matrix) * 100.
    # update peformance meters
    hard_prec.update(hard.item(), input.size(0))
    soft_prec.update(soft.item(), input.size(0))

    if step % args.report_freq == 0:
      logging.info(f'valid step: {step:03d}\t'+
      f'hard prec {hard_prec.val:.3f} ({hard_prec.avg:.3f})\t'+
      f'soft prec {soft_prec.val:.3f} ({soft_prec.avg:.3f})\t')

  return hard_prec, soft_prec


if __name__ == '__main__':
  main() 

