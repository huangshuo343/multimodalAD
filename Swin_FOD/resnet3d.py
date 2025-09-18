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
        self.data = 'all' # t1, taupet, both, all
        self.modality = self.data
        
        self.manual_seed = 42
        self.multistep_milestones = [10, 20]
        self.begin_epoch = 1
        self.n_epochs = 15 # 100
        self.device = device
        self.resume = False
        self.exp_name = f"resnet_3d_random_{self.data}"
        # if resuming, set to the original experiment time
        if self.resume:
            self.exp_time = '2024-12-10_20-44-28'
        else:
            self.exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.output_dir = os.path.join(script_path, "runs", self.exp_name, self.exp_time)
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.output_dir, "model_checkpoint.pth")

if __name__ == '__main__':
    opt = Options()

    # Set random seeds for reproducibility
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    in_channels = 17 if opt.data == 'all' else (2 if opt.data == 'both' else 1)

    # Define the model
    model = generate_model(18, n_input_channels=in_channels, n_classes=3, conv1_t_stride=2).to(opt.device)

    # Define optimizer, scheduler, and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.multistep_milestones)
    criterion = nn.CrossEntropyLoss().to(opt.device)

    tr_data_loader, val_data_loader, test_data_loader = get_loader(opt)

    # Print model summary
    summary(model, (in_channels, 128, 96, 128))

    # Optionally resume training
    if opt.resume:
        opt.begin_epoch = load_model(model, optimizer, opt.checkpoint_path) + 1

    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    # Training loop
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        print(f'Train at epoch {epoch}')

        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()

        for i, all_ in enumerate(tr_data_loader):
            data_time.update(time.time() - end_time)

            if opt.data == 't1' or opt.data == 'taupet':
                inputs, label = all_
            elif opt.data == 'fod':
                fod_o0, fod_o2, fod_o4, label = all_
                inputs = torch.cat([fod_o0, fod_o2, fod_o4], dim=1)
            elif opt.data == 'all':
                t1, taupet_img, fod_o0, fod_o2, fod_o4, label = all_
                inputs = torch.cat([t1, taupet_img, fod_o0, fod_o2, fod_o4], dim=1)
            
            inputs = inputs.to(opt.device)
            label = label.long().to(opt.device)

            _, outputs = model(inputs)
            loss = criterion(outputs, label)
            acc = calculate_accuracy(outputs, label)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if i % 10 == 0:
                print(f'Epoch: [{epoch}][{i + 1}/{len(tr_data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Acc {accuracies.val:.3f} ({accuracies.avg:.3f})')
                # Log to file
                with open(os.path.join(opt.output_dir, "log.txt"),"a") as f:
                    f.write(f'Epoch: [{epoch}][{i + 1}/{len(tr_data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Acc {accuracies.val:.3f} ({accuracies.avg:.3f})\n')

        scheduler.step()

        # validation
        losses = []
        preds = []
        labels = []
        for all_ in val_data_loader:
            if opt.data == 't1' or opt.data == 'taupet':
                inputs, label = all_
            elif opt.data == 'fod':
                fod_o0, fod_o2, fod_o4, label = all_
                inputs = torch.cat([fod_o0, fod_o2, fod_o4], dim=1)
            elif opt.data == 'all':
                t1, taupet_img, fod_o0, fod_o2, fod_o4, label = all_
                inputs = torch.cat([t1, taupet_img, fod_o0, fod_o2, fod_o4], dim=1)

            inputs = inputs.to(opt.device)
            label = label.long().to(opt.device)
            loss, predictions = run_inference(model, inputs, label)
            losses.append(loss)
            preds.extend(predictions)
            labels.extend(label.cpu().numpy())
        loss = np.mean(losses)
        print(f'Validation loss at epoch {epoch}: {loss:.4f}')
        with open(os.path.join(opt.output_dir, "log.txt"),"a") as f:
            f.write(f'Validation loss at epoch {epoch}: {loss:.4f}\n')
        val_accuracy = (np.array(preds) == np.array(labels)).mean()
        print(f'Val accuracy at epoch {epoch}: {val_accuracy:.3f}')
        with open(os.path.join(opt.output_dir, "log.txt"),"a") as f:
            f.write(f'Val accuracy at epoch {epoch}: {val_accuracy:.3f}\n')

        metrics = get_metrics(preds, labels)
        print(f'Validation metrics at epoch {epoch}: {metrics}')
        with open(os.path.join(opt.output_dir, "log.txt"),"a") as f:
            f.write(f'Validation metrics at epoch {epoch}: {metrics}\n')

        if val_accuracy > best_val_accuracy: #loss < best_val_loss
            # best_val_loss = loss
            best_val_accuracy = val_accuracy

            # Save the model at each epoch
            save_model(model, optimizer, epoch, opt.checkpoint_path)

        # Test accuracy
        preds = []
        labels = []
        for all_ in test_data_loader:
            if opt.data == 't1' or opt.data == 'taupet':
                inputs, label = all_
            elif opt.data == 'fod':
                fod_o0, fod_o2, fod_o4, label = all_
                inputs = torch.cat([fod_o0, fod_o2, fod_o4], dim=1)
            elif opt.data == 'all':
                t1, taupet_img, fod_o0, fod_o2, fod_o4, label = all_
                inputs = torch.cat([t1, taupet_img, fod_o0, fod_o2, fod_o4], dim=1)
                
            inputs = inputs.to(opt.device)
            label = label.long().to(opt.device)
            _, predictions = run_inference(model, inputs, label)
            preds.extend(predictions)
            labels.extend(label.cpu().numpy())
        
        test_accuracy = (np.array(preds) == np.array(labels)).mean()
        print(f'Test accuracy at epoch {epoch}: {test_accuracy:.3f}')
        with open(os.path.join(opt.output_dir, "log.txt"),"a") as f:
            f.write(f'Test accuracy at epoch {epoch}: {test_accuracy:.3f}\n')

        metrics = get_metrics(preds, labels)
        print(f'Test metrics at epoch {epoch}: {metrics}')
        with open(os.path.join(opt.output_dir, "log.txt"),"a") as f:
            f.write(f'Test metrics at epoch {epoch}: {metrics}\n')