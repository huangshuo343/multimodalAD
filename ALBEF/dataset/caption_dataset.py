import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            

    
# class pretrain_dataset3d(Dataset):
#     def __init__(self, ann_file, transform_pet, transform_mri):
#         self.ann = []
#         for f in ann_file:
#             self.ann += json.load(open(f,'r'))

#         self.ann = self.ann

#         # self.mris = []
#         # self.tau_pets = []
#         # zoom_factors = [n / old for n, old in zip((128, 96, 128), (192, 128, 192))]

#         # for ann in self.ann:
#         #     tmp = nib.load(ann['mri']).get_fdata().astype(np.float32)
#         #     # crop to 192x128x192
#         #     tmp = tmp[32:224, 32:160, 32:224]
#         #     # resize to 128x96x128
#         #     tmp = zoom(tmp, zoom_factors, order=1)
#         #     self.mris.append(tmp)

#         #     tmp = nib.load(ann['tau_pet']).get_fdata().astype(np.float32)
#         #     # crop to 192x128x192
#         #     tmp = tmp[32:224, 32:160, 32:224]
#         #     # resize to 128x96x128
#         #     tmp = zoom(tmp, zoom_factors, order=1)
#         #     self.tau_pets.append(tmp)
        
#         self.transform_pet = transform_pet
#         self.transform_mri = transform_mri
        
#     def __len__(self):
#         return len(self.ann)

#     def __getitem__(self, index):    
        
#         ann = self.ann[index]

#         mri = nib.load(ann['mri']).get_fdata().astype(np.float32)
#         mri = mri[32:224, 32:160, 32:224]
#         mri = zoom(mri, (128/192, 96/128, 128/192), order=1)[None, ...]
#         tau_pet = nib.load(ann['tau_pet']).get_fdata().astype(np.float32)
#         tau_pet = tau_pet[32:224, 32:160, 32:224]
#         tau_pet = zoom(tau_pet, (128/192, 96/128, 128/192), order=1)[None, ...]
#         inferior_cerebellum = '/'.join(ann['tau_pet'].split('/')[:-1]) + '/km_inferior.ref.tac.dat'
#         inferior_cerebellum = np.loadtxt(inferior_cerebellum).astype(np.float32)
#         tau_pet = tau_pet / inferior_cerebellum
#         # mri = self.mris[index][np.newaxis, ...]
#         # tau_pet = self.tau_pets[index][np.newaxis, ...]
#         # amyloid_pet = nib.load(ann['amyloid_pet']).get_fdata().astype(np.float32)[None, ...]

#         mri = self.transform_mri(mri)
#         tau_pet = self.transform_pet(tau_pet)
#         # amyloid_pet = self.transform_pet(amyloid_pet)
#         # print(type(mri), type(tau_pet))
#         # print(mri.shape, tau_pet.shape)
#         return mri, tau_pet


class pretrain_dataset3d(Dataset):
    def __init__(self, transform_pet, transform_mri, dataset='train'):
        """
        Args:
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        csv_file = '/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/scripts/data_list_MOCAnocopy.csv'
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
                }
            )
        
                    # 'taupet': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/data_to_T1template/braintauPET_to_template/result.nii.gz',
                    # 'fod': f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/ADNI2023use/{sub}/{"".join([year, month, day])}/data_to_T1template/brainT1_to_template/result.1.nii.gz'
        
        train_data, temp_data = train_test_split(self.data, test_size=(1 - 0.6), random_state=42)
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        # data_number = 0
        if dataset == 'train':
            self.data = train_data
            print('train_data:', len(self.data))
            for data_number in range(len(self.data)):
                one_data = train_data[data_number]
                if one_data['label'] == 2:
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

        self.transform_pet = transform_pet
        self.transform_mri = transform_mri

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = self.data[idx]['label']
        t1_path = self.data[idx]['t1']
        taupet_path = self.data[idx]['taupet']
        fod_path = self.data[idx]['fod']

        # Load the NIfTI files
        t1 = nib.load(t1_path).get_fdata().astype(np.float32)  # Load the T1 image
        t1 = (t1 - t1.min()) / (t1.max() - t1.min())
        x, y, z = t1.shape
        t1 = zoom(t1, (128/x, 128/y, 128/z), order=1)
        t1 = t1[None, ...]


        taupet_img = nib.load(taupet_path).get_fdata().astype(np.float32)  # Load the tau PET image
        taupet_img /= 5
        # print('taupet_path', taupet_path, ', taupet_img:', taupet_img.shape)
        # taupet_img = taupet_img[32:224, 32:224, 32:224]
        x, y, z = taupet_img.shape
        taupet_img = zoom(taupet_img, (128/x, 128/y, 128/z), order=1)
        taupet_img = taupet_img[None, ...]


        if self.transform_mri is not None:
            t1 = self.transform_mri(t1)

        if self.transform_pet is not None:
            taupet_img = self.transform_pet(taupet_img)

        return t1, taupet_img, label


class adni_cls_dataset3d(Dataset):
    def __init__(self, ann_file, transform_pet, transform_mri):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        self.ann = self.ann
        
        self.transform_pet = transform_pet
        self.transform_mri = transform_mri
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):    
        
        ann = self.ann[index]

        mri = nib.load(ann['mri']).get_fdata().astype(np.float32)
        mri = mri[32:224, 32:196, 32:224]
        x, y, z = mri.shape
        mri = zoom(mri, (128/x, 96/y, 128/z), order=1)[None, ...]
        # standardize
        mri = (mri - mri.mean()) / mri.std()

        assert mri.shape == (1, 128, 96, 128), f"mri shape is {mri.shape}"

        tau_pet = nib.load(ann['tau_pet']).get_fdata().astype(np.float32)
        tau_pet = tau_pet[32:224, 32:196, 32:224]
        x, y, z = tau_pet.shape
        tau_pet = zoom(tau_pet, (128/x, 96/y, 128/z), order=1)[None, ...]

        assert tau_pet.shape == (1, 128, 96, 128), f"tau_pet shape is {tau_pet.shape}"

        inferior_cerebellum = '/'.join(ann['tau_pet'].split('/')[:-1]) + '/km_inferior.ref.tac.dat'
        inferior_cerebellum = np.loadtxt(inferior_cerebellum).astype(np.float32)
        tau_pet = tau_pet / inferior_cerebellum

        label = float(ann['cdr'])
        label = label if label == 0 else (1 if label == 0.5 else 2)
        # mri = self.mris[index][np.newaxis, ...]
        # tau_pet = self.tau_pets[index][np.newaxis, ...]
        # amyloid_pet = nib.load(ann['amyloid_pet']).get_fdata().astype(np.float32)[None, ...]

        if self.transform_mri is not None:
            mri = self.transform_mri(mri)
        if self.transform_pet is not None:
            tau_pet = self.transform_pet(tau_pet)
        # amyloid_pet = self.transform_pet(amyloid_pet)
        # print(type(mri), type(tau_pet))
        # print(mri.shape, tau_pet.shape)
        return mri, tau_pet, label


import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nb


class MMDataset(Dataset):
    def __init__(self,df,data_name_list):
        # df: df load from csv file
        # data_name_list: list of col names in csv for dadta file loading
        self.df = df
        self.data_name_list = data_name_list
    def __len__(self):
        return self.df['sub'].shape[0]
    
    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        data_list = []
        for i in self.data_name_list:
            if row[i].split('.')[-1] == 'raw':
                data_list.append(np.fromfile(row[i],dtype=np.float32).reshape(1,-1))
            else:
                data_list.append(nb.freesurfer.io.read_morph_data(row[i]).reshape(1,-1))
        x = np.vstack(data_list)
        # split the first 2 and last 2 columns
        mri = x[:2,:]
        pet = x[-2:,:]
        y = int(row['diag_label'])
        return mri, pet, y

