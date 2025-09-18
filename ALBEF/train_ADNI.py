'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchsummary

from models.model_pretrain3D import MultiModal3DClassifier

import utils
from dataset import create_dataset3d, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import warnings

# Filter out the TorchIO warning, where i just use its transform functions and not the full batched pipeline, so it's fine
warnings.filterwarnings(
    'ignore',
    message='Using TorchIO images without a torchio.SubjectsLoader in PyTorch >= 2.3 might have unexpected consequences'
)


def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cls', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    t = time.time()
    for i, (mri, pet, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print('data loading time:', time.time()-t)

        t = time.time()        
        optimizer.zero_grad()
  
        mri = mri.to(device,non_blocking=True) 
        pet = pet.to(device,non_blocking=True)
        label = label.long().to(device,non_blocking=True)

        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        loss_cls, loss_mlm, loss_ita, loss_itm, _ = model(mri, pet, label, alpha)
        loss = loss_cls + loss_mlm + loss_ita + loss_itm

        # print('forward time:', time.time()-t)
          
        t = time.time()
        loss.backward()
        # print('backward time:', time.time()-t)
        
        t = time.time()
        optimizer.step()
        
        
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)
        # print('step time:', time.time()-t)

        t = time.time()
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def validata_and_test(model, data_loader, epoch, device, config, val=True):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_cls', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('real_acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('pre', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('sen', utils.SmoothedValue(window_size=1, fmt='{value:.6f}')) # recall for positive class
    metric_logger.add_meter('spe', utils.SmoothedValue(window_size=1, fmt='{value:.6f}')) # recall for negative class
    metric_logger.add_meter('ma_f1', utils.SmoothedValue(window_size=1, fmt='{value:.6f}')) # average of f1 score for all classes
    metric_logger.add_meter('mi_f1', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Valid Epoch: [{}]'.format(epoch) if val else 'Test Epoch: [{}]'.format(epoch)
    with torch.no_grad():
        for i, (mri, pet, label) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            mri = mri.to(device,non_blocking=True) 
            pet = pet.to(device,non_blocking=True)
            label = label.long().to(device,non_blocking=True)
            loss_cls, loss_mlm, loss_ita, loss_itm, output = model(mri, pet, label, config['alpha'], train=False)
            metric_logger.update(loss_mlm=loss_mlm.item())
            metric_logger.update(loss_ita=loss_ita.item())
            metric_logger.update(loss_itm=loss_itm.item())
            metric_logger.update(loss_cls=loss_cls.item())
            metric_logger.update(real_acc=(torch.argmax(output, dim=1) == label).float().mean().item())
            metric_logger.update(**model.module.get_metrics(output, label))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']    

    #### Dataset #### 
    print("Creating dataset")
    t = time.time()
    datasets = [create_dataset3d('adni_cls_train', config), create_dataset3d('adni_cls_val', config), create_dataset3d('adni_cls_test', config)]
    print('Data loading time:', time.time()-t)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None]

    tr_data_loader, val_data_loader, test_data_loader = create_loader(
        datasets,
        samplers,
        batch_size=[config['batch_size'], config['batch_size'], config['batch_size']], 
        num_workers=[4, 4, 4], 
        is_trains=[True, False, False], 
        collate_fns=[None, None, None])

    #### Model #### 
    print("Creating model")
    model = MultiModal3DClassifier(patch_size=config['patch_size'], config=config, class_num=3)
    
    model = model.to(device)

    # with torch.no_grad():
    #     torchsummary.summary(model, [(1, 128, 96, 128), (1, 128, 96, 128)])
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

    
    assert args.checkpoint != '', 'Please specify the checkpoint to load'

    checkpoint = torch.load(args.checkpoint, map_location='cpu') 
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s'%args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    print("Start training")
    start_time = time.time()
    best_val_loss_cls = float('inf')

    for epoch in range(start_epoch, max_epoch):
        
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
            
        train_stats = train(model, tr_data_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)
        val_stats = validata_and_test(model, val_data_loader, epoch, device, config, val=True)
        if utils.is_main_process():  
            # train
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # validation
            log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            if float(val_stats['loss_cls']) < best_val_loss_cls:

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))

        if float(val_stats['loss_cls']) < best_val_loss_cls:
            best_val_loss_cls = float(val_stats['loss_cls'])
            test_stats = validata_and_test(model, test_data_loader, epoch, device, config, val=False)
            if utils.is_main_process():
                log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")


        dist.barrier()  
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_ADNI.py --config ./configs/train_ADNI.yaml --output_dir output/train_ADNI_bugfixed --checkpoint /ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MICCAI2024_segmentation/ALBEF/output/hable_pretrain_300ep_complex/checkpoint_299.pth
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    # add timestep
    args.output_dir = args.output_dir + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)