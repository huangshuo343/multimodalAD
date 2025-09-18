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
from encoder import generate_model
def get_inplanes():
    # return [64, 128, 256, 512]
    return [16, 32, 64, 128]
    # return [4, 8, 16, 32]

import encoder
encoder.get_inplanes = get_inplanes
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
        self.batch_size = 2
        self.workers = 8
        self.modality = 'all' # t1, taupet
        
        self.manual_seed = 42
        self.multistep_milestones = [10, 20]
        self.begin_epoch = 1
        self.n_epochs = 50
        self.device = device
        self.resume = True
        self.exp_name = f"resnet_3d_random_{self.modality}"
        # if resuming, set to the original experiment time
        if self.modality == 't1':
            self.exp_time = '2025-01-21_15-16-50'
        elif self.modality == 'taupet':
            self.exp_time = '2025-01-21_15-18-14'
        elif self.modality == 'all':
            self.exp_time = '2025-02-16_01-34-21'

        self.output_dir = os.path.join(script_path, "runs", self.exp_name, self.exp_time)
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.output_dir, "model_checkpoint.pth")


# CUDA_VISIBLE_DEVICES=1 python inference_resnet3d.py
if __name__ == '__main__':
    opt = Options()

    # Set random seeds for reproducibility
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    in_channels = 17 if opt.modality == 'all' else 1

    # Define the model
    model = generate_model(18, n_input_channels=in_channels, n_classes=3, conv1_t_stride=2).to(opt.device)

    tr_data_loader, val_data_loader, test_data_loader = get_loader(opt)

    # Print model summary
    summary(model, (in_channels, 128, 96, 128))

    # Optionally resume training
    opt.begin_epoch = load_model(model, None, opt.checkpoint_path) + 1

    rank = 0

    # validation
    test_features = []
    test_features2 = []
    for idx, (t1, taupet_img, fod_o0, fod_o2, fod_o4, label) in enumerate(val_data_loader):

        if rank == 0 and idx % 10 == 0:
            print(f"Validation: {idx}/{len(val_data_loader)}")

        fod_o0, fod_o2, fod_o4, label = fod_o0.cuda(rank), fod_o2.cuda(rank), fod_o4.cuda(rank), label.long().cuda(rank)
        t1, taupet_img = t1.cuda(rank), taupet_img.cuda(rank)
        test_input = torch.cat([t1, taupet_img, fod_o0, fod_o2, fod_o4], dim=1)
        # loss, predictions = run_inference(model, test_input, label)
        t1_features, t1_fc_out = model.get_features(test_input)
        # print(t1_features.shape)
        # print(t1_fc_out.shape)
        # print(label.unsqueeze(1).shape)
        feature = torch.cat([t1_features, label.unsqueeze(1)], dim=1)
        test_features.append(feature.detach().cpu().numpy())

        feature = torch.cat([t1_fc_out, label.unsqueeze(1)], dim=1)
        test_features2.append(feature.detach().cpu().numpy())


    test_features = np.concatenate(test_features, axis=0)
    np.save(os.path.join(script_path, "val_features_resnetall.npy"), test_features)
    test_features2 = np.concatenate(test_features2, axis=0)
    np.save(os.path.join(script_path, "val_features2_resnetall.npy"), test_features2)

    # Test accuracy
    test_features = []
    test_features2 = []
    for idx, (t1, taupet_img, fod_o0, fod_o2, fod_o4, label) in enumerate(test_data_loader):

        if rank == 0 and idx % 10 == 0:
            print(f"Test: {idx}/{len(test_data_loader)}")

        fod_o0, fod_o2, fod_o4, label = fod_o0.cuda(rank), fod_o2.cuda(rank), fod_o4.cuda(rank), label.long().cuda(rank)
        t1, taupet_img = t1.cuda(rank), taupet_img.cuda(rank)
        test_input = torch.cat([t1, taupet_img, fod_o0, fod_o2, fod_o4], dim=1)
        # loss, predictions = run_inference(model, test_input, label)
        t1_features, t1_fc_out = model.get_features(test_input)
        feature = torch.cat([t1_features, label.unsqueeze(1)], dim=1)
        test_features.append(feature.detach().cpu().numpy())

        feature = torch.cat([t1_fc_out, label.unsqueeze(1)], dim=1)
        test_features2.append(feature.detach().cpu().numpy())


    test_features = np.concatenate(test_features, axis=0)
    np.save(os.path.join(script_path, "test_features_resnetall.npy"), test_features)
    test_features2 = np.concatenate(test_features2, axis=0)
    np.save(os.path.join(script_path, "test_features2_resnetall.npy"), test_features2)