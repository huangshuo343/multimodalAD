# %%
import os

path = '/ifs/loni/faculty/shi/spectrum/jiaxiny/HABLE/images'
files = sorted(os.listdir(path))

mri_base = path + "/{sub_id}/T1/{timeline_t1}/{scan_t1}/Freesurfer6/{sub_id}/pet_uniform/amyloid_smoothed_inferior_cerebellum/T1.nii.gz"
amyloid_pet_base = path + "/{sub_id}/T1/{timeline_t1}/{scan_t1}/Freesurfer6/{sub_id}/pet_uniform/amyloid_smoothed_inferior_cerebellum/PET_T1.nii.gz"
tau_pet_base_oneT1scan = path + "/{sub_id}/tauPET/{timeline_tau}/{scan_tau}/tau_smoothed_inferior_cerebellum/PET_T1.nii.gz"
tau_pet_base_mulT1scan = path + "/{sub_id}/tauPET/{timeline_tau}/{scan_tau}/tau_smoothed_inferior_cerebellum_{scan_t1}/PET_T1.nii.gz"

num_sub = 0
subs = []
num_sub_multi_timeline = 0

mris = []
amyloid_pets = []
tau_pets = []

for file in files:
    if not os.path.isdir(os.path.join(path, file)):
        continue
    modals = os.listdir(os.path.join(path, file))
    if ('T1' not in modals) or ('tauPET' not in modals):
        continue
    
    timelines_t1 = os.listdir(os.path.join(path, file, 'T1'))
    timelines_tau = os.listdir(os.path.join(path, file, 'tauPET'))
    
    years_t1 = [timeline[:4] for timeline in timelines_t1]
    years_tau = [timeline[:4] for timeline in timelines_tau]

    # only keep the same year
    years = list(set(years_t1) & set(years_tau))
    if len(years) == 0:
        continue

    if len(years) > 1:
        num_sub_multi_timeline += 1

    timelines_t1 = sorted([timeline for timeline in timelines_t1 if timeline[:4] in years])
    timelines_tau = sorted([timeline for timeline in timelines_tau if timeline[:4] in years])

    if len(timelines_t1) != len(timelines_tau):
        # only 8515 has this case, which has two t1 and one tau
        timelines_t1 = [timelines_t1[-1]]

    appended = False
    for timeline_t1, timeline_tau in zip(timelines_t1, timelines_tau):
        scans_t1 = os.listdir(os.path.join(path, file, 'T1', timeline_t1))
        scans_tau = os.listdir(os.path.join(path, file, 'tauPET', timeline_tau))

        # t1 may have multiple scans, but tau only has one
        if len(scans_t1) == 1:
            scan_t1 = scans_t1[0]
            mri = mri_base.format(sub_id=file, timeline_t1=timeline_t1, scan_t1=scan_t1)
            amyloid_pet = amyloid_pet_base.format(sub_id=file, timeline_t1=timeline_t1, scan_t1=scan_t1)
            tau_pet = tau_pet_base_oneT1scan.format(sub_id=file, timeline_tau=timeline_tau, scan_tau=scans_tau[0])
        else:
            for scan_t1 in scans_t1:
                mri = mri_base.format(sub_id=file, timeline_t1=timeline_t1, scan_t1=scan_t1)
                amyloid_pet = amyloid_pet_base.format(sub_id=file, timeline_t1=timeline_t1, scan_t1=scan_t1)
                tau_pet = tau_pet_base_mulT1scan.format(sub_id=file, timeline_tau=timeline_tau, scan_tau=scans_tau[0], scan_t1=scan_t1)
        
        if not os.path.exists(mri) or not os.path.exists(amyloid_pet) or not os.path.exists(tau_pet):
            continue
        mris.append(mri)
        amyloid_pets.append(amyloid_pet)
        tau_pets.append(tau_pet)
        appended = True

    if appended:
        num_sub += 1
        subs.append(file)

print("subjects:", num_sub, "/", len(files))
print("unique subjects:", len(set(subs)))
print("subjects with multiple timelines:", num_sub_multi_timeline, "/", num_sub)
print("total data:", len(mris), len(amyloid_pets), len(tau_pets))
# %%
for mri, amyloid_pet, tau_pet in zip(mris, amyloid_pets, tau_pets):
    inferior_cerebellum = inferior_cerebellum = '/'.join(tau_pet.split('/')[:-1]) + '/km_inferior.ref.tac.dat'
    assert os.path.exists(inferior_cerebellum), f"{inferior_cerebellum} does not exist"

# %%
import json

data = []
for mri, amyloid_pet, tau_pet in zip(mris, amyloid_pets, tau_pets):
    data.append({"mri": mri, "amyloid_pet": amyloid_pet, "tau_pet": tau_pet})

with open('/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MICCAI2024_segmentation/ALBEF/data/multimodal.json', 'w') as f:
    json.dump(data, f)

# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

data = nib.load('/ifs/loni/faculty/shi/spectrum/jiaxiny/HABLE/images/4106/T1/20230515/I1701456/Freesurfer6/4106/pet_uniform/amyloid_smoothed_inferior_cerebellum/T1.nii.gz').get_fdata().astype(np.float32)

# %%
plt.imshow(data[:, 160, :])