import argparse
import datetime
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import roc_curve
from sklearn.metrics._ranking import roc_auc_score
import timm
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset_chest_xray
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import train_one_epoch, evaluate_chestxray, accuracy
from util.sampler import RASampler
from libauc import losses
from torchvision import models
import timm.optim.optim_factory as optim_factory
from collections import OrderedDict
import segmentation_models_pytorch as smp

from models_mae_cnn import MaskedAutoencoderCNN  # Import your model class

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
    # parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    # parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
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
    # model = smp.__dict__['Unet'](
    #         encoder_name=args.model,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #         encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    #         in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #         classes=3,                      # model output channels (number of classes in your dataset)
    #     )
    model = torch.load(args.finetune, weights_only=False)
    model.eval()
    model.to(args.device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader_test, 10, header):
            # print(batch)
            images, target = batch[0], batch[-1]
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            # print(images.shape, target.shape)
            
            # Use torch.amp.autocast instead of torch.cuda.amp.autocast
            #with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            #    output = model(images)
            #    output=output[1]
            
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(images)
                    output = output[1]
            else:
                # Chạy mà không có autocast nếu không có GPU
                output = model(images)
                output = output[1]

            # # Check if output is a tuple (in case the model returns multiple values)
            # if isinstance(output, tuple):
            #     output = output[0]  # Assume the first element is the main output
            
            output = torch.sigmoid(output)  # Apply sigmoid for multi-label
            all_targets.append(target.cpu())
            all_outputs.append(output.cpu())
    # print(all_outputs)
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    # all_outputs = torch.cat(all_outputs, dim=0).numpy()
    print(all_outputs.shape,all_targets.shape)
    # Compute AUC for each class
    outAUROC = []
    for i in range(5):  # Assuming 5 classes
        fpr, tpr, thresholds = roc_curve(all_targets[:, i], all_outputs[:, i])
        outAUROC.append(roc_auc_score(all_targets[:, i], all_outputs[:, i]))
        
    print(f"AUC avg: {np.mean(outAUROC):.4f}")
    print(f"AUC for each label: {outAUROC}")

    predicted_classes = []
    for i in range(5):
        predicted_class = (all_outputs[:, i] > 0.5).astype(int)
        predicted_classes.append(predicted_class)

    predicted_classes = np.array(predicted_classes).T
    for i in range(5):
        acc = (predicted_classes[:, i] == all_targets[:, i]).mean()
        print(f"Accuracy for disease {i}: {acc:.2f}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
