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

import json
import math
import os

import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom
from skimage.transform import resize

from monai import data, transforms
import torch
import torch.nn.functional as F


from monai.transforms import Compose, RandSpatialCrop, RandFlip, NormalizeIntensity, RandScaleIntensity, RandShiftIntensity, ToTensor
from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

import torch
from torch.utils.data import Dataset
import pandas as pd

class MultiModalDataset(Dataset):
    def __init__(self, csv_file, transform=None, dataset='train', modality='t1'):
        """
        Args:
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_csv(csv_file)
        df = df[['Subject', 'Exam Date', 'Diagnosis']]
        self.data = []
        # Populate the dictionary with subjects as keys
        for sub, date, diagnosis in zip(df['Subject'], df['Exam Date'], df['Diagnosis']):
            label = int(diagnosis)
            year, month, day = date.split("/")
            if len(month) < 2:
                month = "0" + month
            if len(day) < 2:
                day = "0" + day
            self.data.append(
                {
                    'label': label,
                    't1': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/brain.nii.gz',
                    'taupet': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/PET_T1_masked_suvr.nii.gz',
                    # 'fod': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/data_to_T1/FODs_to_template/template_FOD/FOD_volumeX/result.nii.gz'
                    'fod': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/FOD.nii.gz'
                    # 'fod': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/FODL0_to_T1/result.0.nii.gz'
                }
            )
        
                    # 'taupet': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/data_to_T1template/braintauPET_to_template/result.nii.gz',
                    # 'fod': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/data_to_T1template/brainT1_to_template/result.1.nii.gz'
        # not_use_data, use_data = train_test_split(self.data, test_size=(1 - 0.6), random_state=42)
        # train_data, temp_data = train_test_split(use_data, test_size=(1 - 0.6), random_state=42)
        train_data, temp_data = train_test_split(self.data, test_size=(1 - 0.6), random_state=42)
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        # data_number = 0
        if dataset == 'train':
            self.data = train_data
            print('train_data:', len(self.data))
            for data_number in range(len(self.data)):
                one_data = train_data[data_number]
                if one_data['label'] == 2:
                    # continue
                    self.data.append(
                        {
                            'label': one_data['label'],
                            't1': one_data['t1'],
                            'taupet': one_data['taupet'],
                            # 'fod': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/data_to_T1/FODs_to_template/template_FOD/FOD_volumeX/result.nii.gz'
                            'fod': one_data['fod']
                        }
                    )
                elif one_data['label'] == 3:
                    self.data.append(
                        {
                            'label': one_data['label'],
                            't1': one_data['t1'],
                            'taupet': one_data['taupet'],
                            # 'fod': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/data_to_T1/FODs_to_template/template_FOD/FOD_volumeX/result.nii.gz'
                            'fod': one_data['fod']
                        }
                    )
                    self.data.append(
                        {
                            'label': one_data['label'],
                            't1': one_data['t1'],
                            'taupet': one_data['taupet'],
                            # 'fod': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/data_to_T1/FODs_to_template/template_FOD/FOD_volumeX/result.nii.gz'
                            'fod': one_data['fod']
                        }
                    )
                    # continue
                    self.data.append(
                        {
                            'label': one_data['label'],
                            't1': one_data['t1'],
                            'taupet': one_data['taupet'],
                            # 'fod': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/data_to_T1/FODs_to_template/template_FOD/FOD_volumeX/result.nii.gz'
                            'fod': one_data['fod']
                        }
                    )
                # print('data_number:', data_number)
                # data_number = data_number + 1
            # Print class distributions
            data_dataframe = pd.DataFrame(self.data)
            print("Train class distribution:\n", data_dataframe['label'].value_counts(normalize=True))
        elif dataset == 'val':
            self.data = validation_data
            data_dataframe = pd.DataFrame(self.data)
            print("Validation class distribution:\n", data_dataframe['label'].value_counts(normalize=True))
        elif dataset == 'test':
            self.data = test_data
            data_dataframe = pd.DataFrame(self.data)
            print("Test class distribution:\n", data_dataframe['label'].value_counts(normalize=True))

        self.transform = transform
        self.modality = modality

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = self.data[idx]['label']
        t1_path = self.data[idx]['t1']
        taupet_path = self.data[idx]['taupet']
        fod_path = self.data[idx]['fod']

        if self.modality == 't1' or self.modality == 'both' or self.modality == 'all' or self.modality == 'allquick':
            # Load the NIfTI files
            t1 = nib.load(t1_path).get_fdata().astype(np.float32)  # Load the T1 image
            t1 = (t1 - t1.min()) / (t1.max() - t1.min())
            # t1 = t1[32:224, 32:224, 32:224]
            # x, y, z = t1.shape
            # t1 = zoom(t1, (96/x, 96/y, 96/z), order=1)
            t1 = t1[None, ...]

        if self.modality == 'taupet' or self.modality == 'both' or self.modality == 'all' or self.modality == 'allquick':
            taupet_img = nib.load(taupet_path).get_fdata().astype(np.float32)  # Load the tau PET image
            taupet_img /= 5
            # print('taupet_path', taupet_path, ', taupet_img:', taupet_img.shape)
            # taupet_img = taupet_img[32:224, 32:224, 32:224]
            # x, y, z = taupet_img.shape
            # taupet_img = zoom(taupet_img, (96/x, 96/y, 96/z), order=1)
            taupet_img = taupet_img[None, ...]

        if self.modality == 'fod' or self.modality == 'all':
            fod = nib.load(fod_path).get_fdata().astype(np.float32)  # Load the FOD image

            # vols_order0 = []
            # vols_order2 = []
            # vols_order4 = []
            # for volume in range(1):
            #     vol_path = fod_path.replace('volumeX', f'volume{volume}')
            #     vol = nib.load(vol_path).get_fdata()
            #     vol = vol[None, ...]
            #     vols_order0.append(vol)
            # for volume in range(1, 6):
            #     vol_path = fod_path.replace('volumeX', f'volume{volume}')
            #     vol = nib.load(vol_path).get_fdata()
            #     vol = vol[None, ...]
            #     vols_order2.append(vol)
            # for volume in range(6, 15):
            #     vol_path = fod_path.replace('volumeX', f'volume{volume}')
            #     vol = nib.load(vol_path).get_fdata()
            #     vol = vol[None, ...]
            #     vols_order4.append(vol)

            # fod_o0 = np.concatenate(vols_order0, axis=0, dtype=np.float32)
            # fod_o2 = np.concatenate(vols_order2, axis=0, dtype=np.float32)
            # fod_o4 = np.concatenate(vols_order4, axis=0, dtype=np.float32)

            fod_o0 = np.transpose(fod[..., 0:1], (3, 0, 1, 2))
            fod_o2 = np.transpose(fod[..., 1:6], (3, 0, 1, 2))
            fod_o4 = np.transpose(fod[..., 6:15], (3, 0, 1, 2))

            # fod_o0 = zoom(fod_o0, (1., target_shape[0] / fod_o0.shape[1], target_shape[1] / fod_o0.shape[2], target_shape[2] / fod_o0.shape[3]), order=1)
            # fod_o2 = zoom(fod_o2, (1., target_shape[0] / fod_o2.shape[1], target_shape[1] / fod_o2.shape[2], target_shape[2] / fod_o2.shape[3]), order=1)
            # fod_o4 = zoom(fod_o4, (1., target_shape[0] / fod_o4.shape[1], target_shape[1] / fod_o4.shape[2], target_shape[2] / fod_o4.shape[3]), order=1)

            fod_o0 = resize(fod_o0, (1, 256, 256, 256), mode='reflect', anti_aliasing=True)
            fod_o2 = resize(fod_o2, (5, 256, 256, 256), mode='reflect', anti_aliasing=True)
            fod_o4 = resize(fod_o4, (9, 256, 256, 256), mode='reflect', anti_aliasing=True)

            fod_o0 /= fod_o0.std()
            fod_o2 /= fod_o2.std()
            fod_o4 /= fod_o4.std()
        
        if self.modality == 'allquick':
            fod_o0 = nib.load(fod_path).get_fdata().astype(np.float32)  # Load the FOD image
            fod_o0 /= fod_o0.std()
            fod_o0 = fod_o0[None, ...]

        # # Apply transforms if any
        # if self.transform:
        #     t1 = self.transform(t1)
        #     taupet_img = self.transform(taupet_img)
        #     fd = self.transform(fod)

        # data = np.concatenate([t1, taupet_img], axis=0, dtype=np.float32)#
        #data = np.concatenate([t1, fod], axis=0, dtype=np.float32)

        # return t1, taupet_img, fod_o0, fod_o2, fod_o4, label - 1
        # if label == 3:
        #     label = 2

        if self.modality == 't1':
            return t1, label - 1
        
        if self.modality == 'taupet':
            return taupet_img, label - 1
        
        if self.modality == 'both':
            return t1, taupet_img, label - 1

        if self.modality == 'fod':
            return fod_o0, fod_o2, fod_o4, label - 1
        
        if self.modality == 'allquick':
            return t1, taupet_img, fod_o0, label - 1

        if self.modality == 'all':
            return t1, taupet_img, fod_o0, fod_o2, fod_o4, label - 1


def get_loader(args):
    # def generate_random_data(num_samples, image_size, label_size):
    #     """Generate random tensors to mimic image and label data."""
    #     data = []
    #     for _ in range(num_samples):
    #         image = torch.rand(image_size)
    #         label = torch.randint(0, 2, label_size, dtype=torch.long)
    #         data.append({"image": image, "label": label})
    #     return data

    # # Define synthetic data properties
    # num_train_samples = 100  # Number of training samples
    # num_val_samples = 20  # Number of validation samples
    # image_size = (4, args.roi_x, args.roi_y, args.roi_z)  # Example image size
    # label_size = (3, args.roi_x, args.roi_y, args.roi_z)  # Example label size

    # # Generate synthetic datasets
    # train_data = generate_random_data(num_train_samples, image_size, label_size)
    # val_data = generate_random_data(num_val_samples, image_size, label_size)

    # label_path = '/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/scripts/data_list_information.csv'
    label_path = '/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/scripts/data_list_MOCAnocopy.csv'
    train_data = MultiModalDataset(label_path, None, 'train', args.modality)
    val_data = MultiModalDataset(label_path, None, 'val', args.modality)
    test_data = MultiModalDataset(label_path, None, 'test', args.modality)

    # # Define transformations
    # train_transform = Compose(
    #     [
    #         RandSpatialCrop(roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False),
    #         RandFlip(prob=0.5, spatial_axis=0),
    #         RandFlip(prob=0.5, spatial_axis=1),
    #         RandFlip(prob=0.5, spatial_axis=2),
    #         NormalizeIntensity(nonzero=True, channel_wise=True),
    #         RandScaleIntensity(factors=0.1, prob=1.0),
    #         RandShiftIntensity(offsets=0.1, prob=1.0),
    #         ToTensor(),
    #     ]
    # )

    # val_transform = Compose(
    #     [
    #         NormalizeIntensity(nonzero=True, channel_wise=True),
    #         ToTensor(),
    #     ]
    # )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    loader = [train_loader, val_loader, test_loader]

    return loader
