import torch
import torchio as tio
from torch.utils.data import DataLoader
from torchvision import transforms

import random
import numpy as np
import pandas as pd
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset, pretrain_dataset3d, adni_cls_dataset3d, MMDataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import grounding_dataset

from dataset.randaugment import RandomAugment

def create_dataset(dataset, config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform)                  
        return dataset      
               
    elif dataset=='re':          
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset   

    elif dataset=='vqa': 
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train') 
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])       
        return train_dataset, vqa_test_dataset

    elif dataset=='nlvr':   
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset        
               
    elif dataset=='ve':   
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset=='grounding':
        train_transform = transforms.Compose([                        
                transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])         
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')       
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')             
        return train_dataset, test_dataset    
    

class RandomAugment3D:
    def __init__(self, pre_augs, rand_augs, post_augs, num_transforms=2):
        """
        Initialize the 3D Random Augment class.

        Args:
            pre_augs (list): List of TorchIO transformations to apply before random augmentations.
            rand_augs (list): List of TorchIO transformations for random selection.
            post_augs (list): List of TorchIO/PyTorch transformations to apply after augmentations.
            num_transforms (int): Number of random transformations to apply from rand_augs.
        """
        self.pre_augs = pre_augs
        self.rand_augs = rand_augs
        self.post_augs = post_augs
        self.num_transforms = num_transforms

    def __call__(self, image):
        """
        Apply the augmentation pipeline to the input image.

        Args:
            image (TorchIO Subject): The 3D image to transform.

        Returns:
            TorchIO Subject or Tensor: The augmented image.
        """

        selected_augmentations = random.sample(self.rand_augs, self.num_transforms)
        pipeline = tio.Compose(
            self.pre_augs + selected_augmentations + self.post_augs
        )
        return pipeline(image)


def create_dataset3d(dataset, config):

    pretrain_transform_pet = RandomAugment3D(
        pre_augs=[
        ],
        rand_augs=[
            # tio.RandomFlip(axes=(0, 1, 2)),  # Randomly flip along x, y, or z axes
            tio.RandomAffine(scales=(0.8, 1.2), degrees=20),  # Random rotation and scaling
            tio.RandomElasticDeformation(),  # Elastic deformation
            tio.RandomNoise(mean=1, std=0.1),  # Add random Gaussian noise
            tio.Lambda(lambda x: x),  # Identity
        ],
        post_augs=[
        ],
        num_transforms=1
    )

    pretrain_transform_mri = RandomAugment3D(
        pre_augs=[
        ],
        rand_augs=[
            # tio.RandomFlip(axes=(0, 1, 2)),  # Randomly flip along x, y, or z axes
            tio.RandomAffine(scales=(0.8, 1.2), degrees=20),  # Random rotation and scaling
            tio.RandomElasticDeformation(),  # Elastic deformation
            tio.RandomNoise(mean=0, std=0.1),  # Add random Gaussian noise
            tio.RandomMotion(),  # Add random motion artifact
            tio.RandomGhosting(),  # Add random ghosting artifact
            tio.RandomBlur(),  # Add random blur
            tio.RandomSpike(),  # Add random spike
            tio.RandomBiasField(),  # Add random bias field
            tio.Lambda(lambda x: x),  # Identity
        ],
        post_augs=[
        ],
        num_transforms=2
    )
    
    if dataset=='pretrain':
        dataset = pretrain_dataset3d(
            transform_pet=pretrain_transform_pet, 
            transform_mri=pretrain_transform_mri,
            dataset='train'
        )
        return dataset
    
    if dataset=='feat_extract_train':
        dataset = pretrain_dataset3d(
            transform_pet=None, 
            transform_mri=None,
            dataset='train'
        )
        return dataset
    
    if dataset=='feat_extract_val':
        dataset = pretrain_dataset3d(
            transform_pet=None, 
            transform_mri=None,
            dataset='val'
        )
        return dataset

    if dataset=='feat_extract_test':
        dataset = pretrain_dataset3d(
            transform_pet=None, 
            transform_mri=None,
            dataset='test'
        )
        return dataset
    
    if dataset == 'adni_cls_train':
        dataset = adni_cls_dataset3d(
            ann_file=config['train_file'],
            transform_pet=pretrain_transform_pet,
            transform_mri=pretrain_transform_mri
        )
        return dataset
    
    if dataset == 'adni_cls_val':
        dataset = adni_cls_dataset3d(
            ann_file=config['val_file'],
            transform_pet=None,
            transform_mri=None
        )
        return dataset
    
    if dataset == 'adni_cls_test':
        dataset = adni_cls_dataset3d(
            ann_file=config['test_file'],
            transform_pet=None,
            transform_mri=None
        )
        return dataset


def create_dataset_surface(dataset, config):
    dataset = MMDataset(
        df=pd.read_csv(config['train_file']),
        data_name_list=['lh_tau_curv','rh_tau_curv','lh_ct_curv','rh_ct_curv']
    )
    return dataset
    
def create_dataset_surfacetest(dataset, config):
    dataset = MMDataset(
        df=pd.read_csv(config['test_file']),
        data_name_list=['lh_tau_curv','rh_tau_curv','lh_ct_curv','rh_ct_curv']
    )
    return dataset
    

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    