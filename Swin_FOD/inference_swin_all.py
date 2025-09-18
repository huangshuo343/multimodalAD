# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import time
import numpy as np
import torch
from torchsummary import summary
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.data_utils import get_loader

from monai.metrics import ConfusionMatrixMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

from model import MySwinUNETR

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline for BRATS Challenge")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--json_list", default="./jsons/brats21_folds.json", type=str, help="dataset json file")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
parser.add_argument("--modality", default="all", type=str, help="modality for training")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=24, type=int, help="feature size")
parser.add_argument("--in_channels", default=17, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=4, type=int, help="number of output channels")
parser.add_argument("--cache_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=32, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=32, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=32, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument(
    "--pretrained_dir",
    default="./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/",
    type=str,
    help="pretrained checkpoint directory",
)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
import time
import os
from datetime import datetime
from torchsummary import summary
from encoder import generate_model, generate_model_nonfod
from utils.data_utils import get_loader

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_path = os.path.dirname(os.path.abspath(__file__))

# Helper classes for tracking metrics
class AverageMeter:
    """Tracks and computes the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Placeholder for accuracy calculation
def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

# Function to save the model
def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Model saved to {path}")

# Function to load the model
def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {path}, starting from epoch {epoch}")
    return epoch

# Function for inference
def run_inference(model, input_tensor, label):
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        _, outputs = model(input_tensor)
        loss = F.cross_entropy(outputs, label)
        _, preds = torch.max(outputs, 1)
    return loss.cpu().numpy(), preds.cpu().numpy()

def get_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    # Initialize metrics
    tp = np.zeros(3)  # True Positives for each class
    fp = np.zeros(3)  # False Positives for each class
    fn = np.zeros(3)  # False Negatives for each class
    tn = np.zeros(3)  # True Negatives for each class

    for i in range(3):  # Iterate over each class
        tp[i] = ((preds == i) & (labels == i)).sum().item()
        fp[i] = ((preds == i) & (labels != i)).sum().item()
        fn[i] = ((preds != i) & (labels == i)).sum().item()
        tn[i] = ((preds != i) & (labels != i)).sum().item()

    # Compute metrics per class
    precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-8)  # Sensitivity
    specificity = tn / (tn + fp + 1e-8)  # Specificity
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Aggregate metrics
    accuracy = (tp.sum() + tn.sum()) / (tp.sum() + fp.sum() + fn.sum() + tn.sum())
    macro_f1 = f1_scores.mean()  # Average F1 in this case
    micro_precision = tp.sum() / (tp.sum() + fp.sum() + 1e-8)
    micro_recall = tp.sum() / (tp.sum() + fn.sum() + 1e-8)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-8)

    return {
        'acc': accuracy,
        'pre': precision,
        'sen': recall,
        'spe': specificity,
        'ma_f1': macro_f1,
        'mi_f1': micro_f1
    }

# Define options (replace with argument parsing if needed)
class Options:
    def __init__(self):
        self.batch_size = 4
        self.workers = 8
        self.modality = 'all' # t1, taupet
        
        self.manual_seed = 42
        self.multistep_milestones = [10, 20]
        self.begin_epoch = 1
        self.n_epochs = 50
        self.device = device
        self.resume = True
        self.exp_name = f"resnet_3d_{self.modality}"
        # if resuming, set to the original experiment time



# conda activate multimodal
# cd /ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/swinad/
# CUDA_VISIBLE_DEVICES=5 python inference_swin_all.py --batch_size=1 --optim_lr=1e-4 --lrschedule=warmup_cosine --save_checkpoint --noamp --pretrained_dir ./runs/unetr_test_random__all_20250206-164528
# --pretrained_dir ./runs/unetr_test_20250123-190121
if __name__ == "__main__":
    args = parser.parse_args()
    args.amp = not args.noamp

    loader = get_loader(args)
    
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    fod_model = MySwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        modality=args.modality,
    ).to('cuda')

    # summary(model, (args.in_channels, 256, 256, 256))

    model_dict = torch.load(pretrained_pth)["state_dict"]
    fod_model.load_state_dict(model_dict)
    print("Using pretrained weights")

    train_loader=loader[0]
    val_loader=loader[1]
    test_loader=loader[2]

    opt = Options()

    # Set random seeds for reproducibility
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    in_channels = 1

    # Define the model
    # t1_model = generate_model_nonfod(18, n_input_channels=1, n_classes=3, conv1_t_stride=2).to(opt.device)
    # tau_model = generate_model_nonfod(18, n_input_channels=1, n_classes=3, conv1_t_stride=2).to(opt.device)

    # # Print model summary
    # summary(model, (in_channels, 128, 96, 128))

    # t1_path = os.path.join(script_path, "runs", "resnet_3d_t1", '2025-01-21_15-16-50', "model_checkpoint.pth")
    # tau_path = os.path.join(script_path, "runs", "resnet_3d_taupet", '2025-01-21_15-18-14', "model_checkpoint.pth")
    # t1_path = os.path.join(script_path, "runs", "resnet_3d_random_t1", '2025-02-06_14-08-29', "model_checkpoint.pth")
    # tau_path = os.path.join(script_path, "runs", "resnet_3d_random_taupet", '2025-02-06_14-13-30', "model_checkpoint.pth")

    # Optionally resume training
    # load_model(t1_model, None, t1_path)
    # load_model(tau_model, None, tau_path)

    # train set
    train_features = []
    train_features2 = []
    # for idx, (t1, taupet_img, fod_o0, fod_o2, fod_o4, label) in enumerate(train_loader):

    #     if args.rank == 0 and idx % 10 == 0:
    #         print(f"Train: {idx}/{len(train_loader)}")

    #     fod_o0, fod_o2, fod_o4, label = fod_o0.cuda(args.rank), fod_o2.cuda(args.rank), fod_o4.cuda(args.rank), label.long().cuda(args.rank)
    #     t1, taupet_img = t1.cuda(args.rank), taupet_img.cuda(args.rank)

    #     fod_features = fod_model.get_features(t1, taupet_img, fod_o0, fod_o2, fod_o4)
    #     # fod_features = torch.mean(fod_features, dim=(2, 3, 4))
    #     fod_feature1, fod_feature2 = fod_features

    #     # t1_features, t1_fc_out = t1_model.get_features(t1)

    #     # taupet_features, tau_fc_out = tau_model.get_features(taupet_img)

    #     # shape: torch.Size([1, 64]) torch.Size([1, 128]) torch.Size([1, 128]) torch.Size([1, 1])
    #     # torch.Size([1, 3]) torch.Size([1, 3]) torch.Size([1, 3]) torch.Size([1, 1])

    #     # print(fod_feature1.shape, t1_features.shape, taupet_features.shape, label[None,:].shape)
    #     # print(fod_feature2.shape, t1_fc_out.shape, tau_fc_out.shape, label[None,:].shape)

    #     feature = torch.cat([fod_feature1, label[None,:]], dim=1)
    #     train_features.append(feature.detach().cpu().numpy())

    #     feature = torch.cat([fod_feature2, label[None,:]], dim=1)
    #     train_features2.append(feature.detach().cpu().numpy())

    # train_features = np.concatenate(train_features, axis=0)
    # train_features2 = np.concatenate(train_features2, axis=0)
    # # save features
    # np.save(os.path.join(script_path, "train_features_random.npy"), train_features)
    # np.save(os.path.join(script_path, "train_features2_random.npy"), train_features2)


    # validation
    val_features = []
    val_features2 = []
    for idx, (t1, taupet_img, fod_o0, fod_o2, fod_o4, label) in enumerate(val_loader):

        if args.rank == 0 and idx % 10 == 0:
            print(f"Validation: {idx}/{len(val_loader)}")

        fod_o0, fod_o2, fod_o4, label = fod_o0.cuda(args.rank), fod_o2.cuda(args.rank), fod_o4.cuda(args.rank), label.long().cuda(args.rank)
        t1, taupet_img = t1.cuda(args.rank), taupet_img.cuda(args.rank)

        fod_features = fod_model.get_features_all(t1, taupet_img, fod_o0, fod_o2, fod_o4)
        # fod_features = torch.mean(fod_features, dim=(2, 3, 4))
        # print(fod_features[0].shape, fod_features[1].shape)
        fod_feature1, fod_feature2 = fod_features

        # t1_features, t1_fc_out = t1_model.get_features(t1)

        # taupet_features, tau_fc_out = tau_model.get_features(taupet_img)

        feature = torch.cat([fod_feature1, label[None,:]], dim=1)
        val_features.append(feature.detach().cpu().numpy())

        feature = torch.cat([fod_feature2, label[None,:]], dim=1)
        print('feature', feature.shape)
        val_features2.append(feature.detach().cpu().numpy())

    val_features = np.concatenate(val_features, axis=0)
    np.save(os.path.join(script_path, "val_features_randomall.npy"), val_features)
    val_features2 = np.concatenate(val_features2, axis=0)
    np.save(os.path.join(script_path, "val_features2_randomall.npy"), val_features2)
    
    # Test accuracy
    test_features = []
    test_features2 = []
    for idx, (t1, taupet_img, fod_o0, fod_o2, fod_o4, label) in enumerate(test_loader):

        if args.rank == 0 and idx % 10 == 0:
            print(f"Test: {idx}/{len(test_loader)}")

        fod_o0, fod_o2, fod_o4, label = fod_o0.cuda(args.rank), fod_o2.cuda(args.rank), fod_o4.cuda(args.rank), label.long().cuda(args.rank)
        t1, taupet_img = t1.cuda(args.rank), taupet_img.cuda(args.rank)

        fod_features = fod_model.get_features_all(t1, taupet_img, fod_o0, fod_o2, fod_o4)
        # fod_features = torch.mean(fod_features, dim=(2, 3, 4))
        fod_feature1, fod_feature2 = fod_features

        # t1_features, t1_fc_out = t1_model.get_features(t1)

        # taupet_features, tau_fc_out = tau_model.get_features(taupet_img)

        feature = torch.cat([fod_feature1, label[None,:]], dim=1)
        test_features.append(feature.detach().cpu().numpy())

        feature = torch.cat([fod_feature2, label[None,:]], dim=1)
        test_features2.append(feature.detach().cpu().numpy())


    test_features = np.concatenate(test_features, axis=0)
    np.save(os.path.join(script_path, "test_features_randomall.npy"), test_features)
    test_features2 = np.concatenate(test_features2, axis=0)
    np.save(os.path.join(script_path, "test_features2_randomall.npy"), test_features2)
    
    # test_accuracy = (np.array(preds) == np.array(labels)).mean()
    # print(f'Test accuracy: {test_accuracy:.3f}')

    # metrics = get_metrics(preds, labels)
    # print(f'Test metrics: {metrics}')