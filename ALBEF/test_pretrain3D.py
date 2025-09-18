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

import torchsummary

from models.model_pretrain3D import ALBEF3D

import utils
from dataset import create_dataset3d, create_sampler, create_loader

import warnings

# Filter out the TorchIO warning, where i just use its transform functions and not the full batched pipeline, so it's fine
warnings.filterwarnings(
    'ignore',
    message='Using TorchIO images without a torchio.SubjectsLoader in PyTorch >= 2.3 might have unexpected consequences'
)


def train(model, data_loader, device, config):
    model.eval()
    
    all_aligned = []
    all_fusioned = []
    labels = []

    with torch.no_grad():
        for i, (mri, pet, label) in enumerate(data_loader):
    
            mri = mri.to(device,non_blocking=True) 
            pet = pet.to(device,non_blocking=True)
            label = label.long().to(device,non_blocking=True)
            
            mri_embeds, pet_embeds, fusioned = model.feature_extract(mri, pet)

            all_aligned.append(np.concatenate([mri_embeds[:, 0, :].cpu().numpy(), pet_embeds[:, 0, :].cpu().numpy()], 1))
            all_fusioned.append(fusioned[:, 0, :].cpu().numpy())
            labels.append(label.cpu().numpy())

            if i % 10 == 0:
                print("tr", i)

    all_aligned = np.concatenate(all_aligned, 0)
    all_fusioned = np.concatenate(all_fusioned, 0)
    labels = np.concatenate(labels, 0)

    print(all_aligned.shape)
    print(all_fusioned.shape)
    print(labels.shape)

    return all_aligned, all_fusioned, labels


def validata_and_test(model, data_loader, device, config):
    model.eval()

    all_aligned = []
    all_fusioned = []
    labels = []

    with torch.no_grad():
        for i, (mri, pet, label) in enumerate(data_loader):
            mri = mri.to(device,non_blocking=True) 
            pet = pet.to(device,non_blocking=True)
            label = label.long().to(device,non_blocking=True)
            
            mri_embeds, pet_embeds, fusioned = model.feature_extract(mri, pet)

            all_aligned.append(np.concatenate([mri_embeds[:, 0, :].cpu().numpy(), pet_embeds[:, 0, :].cpu().numpy()], 1))
            all_fusioned.append(fusioned[:, 0, :].cpu().numpy())
            labels.append(label.cpu().numpy())

            if i % 10 == 0:
                print("val/test", i)

    all_aligned = np.concatenate(all_aligned, 0)
    all_fusioned = np.concatenate(all_fusioned, 0)
    labels = np.concatenate(labels, 0)

    print(all_aligned.shape)
    print(all_fusioned.shape)
    print(labels.shape)

    return all_aligned, all_fusioned, labels
    

def main(args, config):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating dataset")
    t = time.time()
    datasets = [create_dataset3d('feat_extract_train', config), create_dataset3d('feat_extract_val', config), create_dataset3d('feat_extract_test', config)]
    print('Data loading time:', time.time()-t)
    
    samplers = [None,None,None]

    tr_data_loader, val_data_loader, test_data_loader = create_loader(
        datasets,
        samplers,
        batch_size=[1, 1, 1], 
        num_workers=[4, 4, 4], 
        is_trains=[False, False, False], 
        collate_fns=[None, None, None])

    #### Model #### 
    print("Creating model")
    model = ALBEF3D(patch_size=config['patch_size'], config=config)
    
    model = model.to(device)

    assert args.checkpoint != '', 'Please specify the checkpoint to load'

    checkpoint = torch.load(args.checkpoint, map_location='cpu') 
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s'%args.checkpoint)

    model_without_ddp = model

    tr_all_aligned, tr_all_fusioned, tr_labels = train(model, tr_data_loader, device, config)
    # save features separately
    np.savez(os.path.join(args.output_dir, 'tr_feat_aligned.npz'), aligned=tr_all_aligned)
    np.savez(os.path.join(args.output_dir, 'tr_feat_fusioned.npz'), fusioned=tr_all_fusioned)
    np.savez(os.path.join(args.output_dir, 'tr_label.npz'), label=tr_labels)
    
    val_all_aligned, val_all_fusioned, val_labels = validata_and_test(model, val_data_loader, device, config)
    # save features
    np.savez(os.path.join(args.output_dir, 'val_feat_aligned.npz'), aligned=val_all_aligned)
    np.savez(os.path.join(args.output_dir, 'val_feat_fusioned.npz'), fusioned=val_all_fusioned)
    np.savez(os.path.join(args.output_dir, 'val_label.npz'), label=val_labels)

    test_all_aligned, test_all_fusioned, test_labels = validata_and_test(model, test_data_loader, device, config)
    # save features
    np.savez(os.path.join(args.output_dir, 'test_feat_aligned.npz'), aligned=test_all_aligned)
    np.savez(os.path.join(args.output_dir, 'test_feat_fusioned.npz'), fusioned=test_all_fusioned)
    np.savez(os.path.join(args.output_dir, 'test_label.npz'), label=test_labels)

            
# CUDA_VISIBLE_DEVICES=1 python test_pretrain3D.py --config ./configs/test_pretrain3D.yaml --output_dir output/test_pretrain --checkpoint /ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/ALBEF/output/Pretrain3D/checkpoint_299.pth
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_pretrain3D.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # add timestep
    args.output_dir = args.output_dir + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    print("what the fuck")

    main(args, config)