import argparse
import datetime
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import roc_curve

import timm
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset_chest_xray
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit
from engine_finetune import train_one_epoch, evaluate_chestxray, accuracy
from util.sampler import RASampler
from libauc import losses
from torchvision import models
import timm.optim.optim_factory as optim_factory
from collections import OrderedDict


def get_args_parser():
    parser = argparse.ArgumentParser('Brute Force for image classification', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    # parser.add_argument('--epochs', default=50, type=int)
    # parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true', default=True, help='Use global pool for classification')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool')
    parser.add_argument('--data_path', default='data/CheXpert-v1.0/', type=str, help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int, help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--vit_dropout_rate', type=float, default=0, help='Dropout rate for ViT blocks')
    parser.add_argument("--build_timm_transform", action='store_true', default=False)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True, help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--train_list", default=None, type=str, help="file for train list")
    parser.add_argument("--val_list", default=None, type=str, help="file for val list")
    parser.add_argument("--test_list", default=None, type=str, help="file for test list")
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument("--dataset", default='chexpert', type=str)
    parser.add_argument("--norm_stats", default=None, type=str)
    parser.add_argument('--loss_func', default=None, type=str)
    parser.add_argument("--checkpoint_type", default=None, type=str)
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(f"{args}".replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_test = build_dataset_chest_xray(split='test', args=args)
    sampler_test = SequentialSampler(dataset_test)
    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models.__dict__[args.model](num_classes=args.nb_classes)
    checkpoint = torch.load(args.finetune, map_location=args.device)
    if 'state_dict' in checkpoint.keys():
        checkpoint_model = checkpoint['state_dict']
    elif 'model' in checkpoint.keys():
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint
    msg = model.load_state_dict(checkpoint_model, strict=False)

    model.eval()
    model.to(device)
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    thresholds = torch.linspace(0, 1, 1000)
    all_targets = []
    all_outputs = []

    # with torch.no_grad():
    #     for batch in metric_logger.log_every(data_loader_test, 10, header):
    #         images, target = batch[0], batch[-1]
    #         images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
    #         # output = model(images)
    #         with torch.cuda.amp.autocast():
    #             output = model(images)
    #         output = torch.sigmoid(output)
    #         # print(output:.4f,"/n")
    #         all_targets.append(target.cpu())
    #         all_outputs.append(output.cpu())

    # all_targets = torch.cat(all_targets, dim=0).numpy()
    # all_outputs = torch.cat(all_outputs, dim=0).numpy()
    # print(all_outputs)

    # for i in range(5):
    #     fpr, tpr, thresholds = roc_curve(all_targets[:, i], all_outputs[:, i])
    #     dist = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    #     closest_idx = np.argmin(dist)
    #     threshold = thresholds[closest_idx]
    #     print(f"For label {i}, the threshold closest to (0, 1) is: {threshold:.4f}")
    
    all_targets1 = []
    all_outputs1 = []

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader_test, 10, header):
            images, target1 = batch[0], batch[-1]
            images, target1 = images.to(device, non_blocking=True), target1.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                output1 = model(images)
            output1 = torch.sigmoid(output1)
            all_targets1.append(target1.cpu())
            all_outputs1.append(output1.cpu())

    all_targets1 = torch.cat(all_targets1, dim=0).numpy()
    all_outputs1 = torch.cat(all_outputs1, dim=0).numpy()
    # threshold = [0.098, 0.311, 0.056, 0.311, 0.479]
    threshold = [0.3899, 0.0, 0.0166, 0.0, 0.0033] 
    print(f"Threshold: ",threshold)

    predicted_classes = []
    for i in range(5):
        predicted_class = (all_outputs1[:, i] > threshold[i]).astype(int)
        predicted_classes.append(predicted_class)

    predicted_classes = np.array(predicted_classes).T
    for i in range(5):
        acc = (predicted_classes[:, i] == all_targets1[:, i]).mean()
        print(f"Accuracy for disease {i}: {acc:.2f}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
