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

import os
import pdb
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_fscore_support

import json

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

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, all_ in enumerate(loader):
        if args.modality == 't1' or args.modality == 'taupet':
            raise NotImplementedError
        elif args.modality == 'fod':
            fod_o0, fod_o2, fod_o4, target = all_
            fod_o0, fod_o2, fod_o4, target = fod_o0.cuda(args.rank), fod_o2.cuda(args.rank), fod_o4.cuda(args.rank), target.cuda(args.rank)
        elif args.modality == 'all':
            t1, taupet_img, fod_o0, fod_o2, fod_o4, target = all_
            fod_o0, fod_o2, fod_o4, target = fod_o0.cuda(args.rank), fod_o2.cuda(args.rank), fod_o4.cuda(args.rank), target.cuda(args.rank)
            t1, taupet_img = t1.cuda(args.rank), taupet_img.cuda(args.rank)
        # print("here0")
        # t1, taupet_img, fod_o0, fod_o2, fod_o4, target = t1.cuda(args.rank), taupet_img.cuda(args.rank), fod_o0.cuda(args.rank), fod_o2.cuda(args.rank), fod_o4.cuda(args.rank), target.cuda(args.rank)
        

        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=args.amp):
            # logits, logits_t1, logits_tau, logits_fod_o0, logits_fod_o2, logits_fod_o4 = model(t1, taupet_img, fod_o0, fod_o2, fod_o4)
            if args.modality == 'all':
                logits, logits_t1, logits_tau, logits_fod_o0, logits_fod_o2, logits_fod_o4 = model(t1=t1, taupet_img=taupet_img, fod_o0=fod_o0, fod_o2=fod_o2, fod_o4=fod_o4)

                loss2 = loss_func(logits_t1, target)
                loss3 = loss_func(logits_tau, target)
            else:
                logits, logits_fod_o0, logits_fod_o2, logits_fod_o4 = model(fod_o0=fod_o0, fod_o2=fod_o2, fod_o4=fod_o4)

            # print("here0.1")
            # print(logits.shape, target.shape)
            loss1 = loss_func(logits, target)
            
            loss4 = loss_func(logits_fod_o0, target)
            loss5 = loss_func(logits_fod_o2, target)
            loss6 = loss_func(logits_fod_o4, target)

            loss = loss1 + loss4 + loss5 + loss6

        # print("here1")
        
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # print("here1.1")
            loss.backward()
            # print("here1.2")
            optimizer.step()
        
        # print("here2")

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        
        # print("here3")

        if args.rank == 0 and idx % 10 == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()

    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, post_label=None, post_sigmoid=None, post_pred=None, loss_func=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    all_preds = []
    all_probs = []
    all_labels = []
    all_loss = []

    with torch.no_grad():
        for idx, all_ in enumerate(loader):
            # fod_o0, fod_o2, fod_o4, target = fod_o0.cuda(args.rank), fod_o2.cuda(args.rank), fod_o4.cuda(args.rank), target.cuda(args.rank)
            if args.modality == 't1' or args.modality == 'taupet':
                raise NotImplementedError
            elif args.modality == 'fod':
                fod_o0, fod_o2, fod_o4, target = all_
                fod_o0, fod_o2, fod_o4, target = fod_o0.cuda(args.rank), fod_o2.cuda(args.rank), fod_o4.cuda(args.rank), target.cuda(args.rank)

                with torch.amp.autocast('cuda', enabled=args.amp):
                    logits, logits_fod_o0, logits_fod_o2, logits_fod_o4 = model(fod_o0=fod_o0, fod_o2=fod_o2, fod_o4=fod_o4)
            elif args.modality == 'all':
                t1, taupet_img, fod_o0, fod_o2, fod_o4, target = all_
                fod_o0, fod_o2, fod_o4, target = fod_o0.cuda(args.rank), fod_o2.cuda(args.rank), fod_o4.cuda(args.rank), target.cuda(args.rank)
                t1, taupet_img = t1.cuda(args.rank), taupet_img.cuda(args.rank)

                with torch.amp.autocast('cuda', enabled=args.amp):
                    logits, logits_t1, logits_tau, logits_fod_o0, logits_fod_o2, logits_fod_o4 = model(t1=t1, taupet_img=taupet_img, fod_o0=fod_o0, fod_o2=fod_o2, fod_o4=fod_o4)

            

            loss1 = loss_func(logits, target)
            if args.modality == 'all':
                loss2 = loss_func(logits_t1, target)
                loss3 = loss_func(logits_tau, target)
            loss4 = loss_func(logits_fod_o0, target)
            loss5 = loss_func(logits_fod_o2, target)
            loss6 = loss_func(logits_fod_o4, target)

            loss = loss1 + loss4 + loss5 + loss6 + (loss2 + loss3 if args.modality == 'all' else 0)
            all_loss.append(loss.item())
                
            # Compute accuracy for this batch
            preds = torch.argmax(logits, dim=1)  # Get predicted class indices
            probs = torch.softmax(logits, dim=1)
            correct = (preds == target).sum()  # Count correct predictions
            total = target.size(0)  # Total number of samples in the batch
            acc = correct / total  # Accuracy for this batch

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            # # compute accuracy for t1 and tau
            # preds_t1 = torch.argmax(logits_t1, dim=1)
            # correct_t1 = (preds_t1 == target).sum()
            # acc_t1 = correct_t1 / total

            # preds_tau = torch.argmax(logits_tau, dim=1)
            # correct_tau = (preds_tau == target).sum()
            # acc_tau = correct_tau / total

            # compute accuracy for fod_o0, fod_o2, fod_o4
            preds_fod_o0 = torch.argmax(logits_fod_o0, dim=1)
            correct_fod_o0 = (preds_fod_o0 == target).sum()
            acc_fod_o0 = correct_fod_o0 / total

            preds_fod_o2 = torch.argmax(logits_fod_o2, dim=1)
            correct_fod_o2 = (preds_fod_o2 == target).sum()
            acc_fod_o2 = correct_fod_o2 / total

            preds_fod_o4 = torch.argmax(logits_fod_o4, dim=1)
            correct_fod_o4 = (preds_fod_o4 == target).sum()
            acc_fod_o4 = correct_fod_o4 / total

            # print(f"Batch {idx}: Accuracy = {acc * 100:.2f}%, T1 Accuracy = {acc_t1 * 100:.2f}%, Tau Accuracy = {acc_tau * 100:.2f}%")
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, total*args.world_size], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=total)

            if args.rank == 0:
                acc = run_acc.avg
                # print(
                #     "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                #     ", acc:",
                #     acc,
                #     ", time {:.2f}s".format(time.time() - start_time),
                # )
            start_time = time.time()

    return np.mean(all_loss), run_acc.avg, all_preds, all_labels, all_probs


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_sigmoid=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = torch.amp.GradScaler('cuda')
    val_loss_max = float("inf")
    final_test_acc = 0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            # print to txt file
            with open(os.path.join(args.logdir, "train_log.txt"), "a") as f:
                f.write(
                    "Epoch {}/{}: loss: {:.4f}, time {:.2f}s\n".format(
                        epoch, args.max_epochs - 1, train_loss, time.time() - epoch_time
                    )
                )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_loss, val_acc, val_preds, val_labels, val_probs = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                args=args,
                post_label=post_label,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
                loss_func=loss_func,
            )

            if args.rank == 0:
                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", val_acc:",
                    val_acc,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )
                # print to txt file
                with open(os.path.join(args.logdir, "train_log.txt"), "a") as f:
                    f.write(
                        "Epoch {}/{}: val_acc: {:.4f}, time {:.2f}s\n".format(
                            epoch, args.max_epochs - 1, val_acc, time.time() - epoch_time
                        )
                    )

                preds = np.array(val_preds)
                labels = np.array(val_labels)
                probs = np.array(val_probs)

                print(labels, probs)
                print(labels.shape, probs.shape)

                # Assume y_true and y_pred (predicted labels) are numpy arrays.
                acc = accuracy_score(labels, preds)
                # For ROC AUC, if binary classification, pass probability scores; for multi-class use multi_class='ovr' or 'ovo'
                roc_auc = roc_auc_score(labels, probs, multi_class='ovr')  
                # prec = precision_score(labels, preds, average='weighted')
                # rec = recall_score(labels, preds, average='weighted')
                # f1 = f1_score(labels, preds, average='weighted')
                mcc = matthews_corrcoef(labels, preds)

                # Confusion matrix
                # conf_matrix = np.array([[209,  34,   0],
                #                         [ 49,  87,   7],
                #                         [  2,  28,  32]])
                # conf_matrix = confusion_matrix(y_test, predictions)

                # Extracting true labels and predicted labels
                # y_true = y_test # np.repeat(np.arange(3), conf_matrix.sum(axis=1))
                # y_pred = predictions # np.concatenate([np.repeat(i, conf_matrix[i, :].sum()) for i in range(3)])

                # Compute Precision, Recall, F1-score per class
                precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
                prec=np.mean(precision)
                rec=np.mean(recall)
                f1=np.mean(f1)

                # Compute Balanced Accuracy
                # balanced_acc = balanced_accuracy_score(labels, prays)

                # Compute Matthews Correlation Coefficient (MCC)
                # mcc = matthews_corrcoef(labels, preds)

                metrics = {
                    'accuracy': acc,
                    'roc_auc': roc_auc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'matthews_corrcoef': mcc
                }

                # Write metrics into a log file (appending to it)
                with open(os.path.join(args.logdir, "train_log.txt"), "a") as f:
                    f.write(json.dumps({"epoch": epoch, "val_metrics": metrics}) + "\n")

                if writer is not None:
                    writer.add_scalar("val_acc", val_acc, epoch)

                val_avg_acc = val_acc
                if val_loss > 0: #val_loss < val_loss_max
                    val_metrics = get_metrics(val_preds, val_labels)

                    _, test_acc, test_preds, test_labels, test_probs = val_epoch(
                        model,
                        test_loader,
                        epoch=epoch,
                        acc_func=acc_func,
                        args=args,
                        post_label=post_label,
                        post_sigmoid=post_sigmoid,
                        post_pred=post_pred,
                        loss_func=loss_func,
                    )
                    final_test_acc = test_acc
                    test_metrics = get_metrics(test_preds, test_labels)

                    print("Validation accuracy: ", val_avg_acc)
                    print("Validation loss: ", val_loss)
                    print("Validation metrics: ", val_metrics)
                    print("Test accuracy: ", test_acc)
                    print("Test metrics: ", test_metrics)
                    # print to txt file
                    with open(os.path.join(args.logdir, "train_log.txt"), "a") as f:
                        f.write("Validation accuracy: {:.4f}\n".format(val_avg_acc))
                        f.write("Validation metrics: {}\n".format(val_metrics))
                        f.write("Test accuracy: {:.4f}\n".format(test_acc))
                        f.write("Test metrics: {}\n".format(test_metrics))

                    preds = np.array(test_preds)
                    labels = np.array(test_labels)
                    probs = np.array(test_probs)

                    print(labels, probs)
                    print(labels.shape, probs.shape)

                    # Assume y_true and y_pred (predicted labels) are numpy arrays.
                    acc = accuracy_score(labels, preds)
                    # For ROC AUC, if binary classification, pass probability scores; for multi-class use multi_class='ovr' or 'ovo'
                    roc_auc = roc_auc_score(labels, probs, multi_class='ovr')  
                    # prec = precision_score(labels, preds, average='weighted')
                    # rec = recall_score(labels, preds, average='weighted')
                    # f1 = f1_score(labels, preds, average='weighted')
                    mcc = matthews_corrcoef(labels, preds)

                    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
                    prec=np.mean(precision)
                    rec=np.mean(recall)
                    f1=np.mean(f1)

                    metrics = {
                        'accuracy': acc,
                        'roc_auc': roc_auc,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'matthews_corrcoef': mcc
                    }

                    # Write metrics into a log file (appending to it)
                    with open(os.path.join(args.logdir, "train_log.txt"), "a") as f:
                        f.write(json.dumps({"epoch": epoch, "test_metrics": metrics}) + "\n")



                    val_loss_max = val_loss
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=final_test_acc, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", final_test_acc)

    return final_test_acc
