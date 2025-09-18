# %%
import os

path = '/scratch/faculty/shi/spectrum/jiaxiny/Datasets/ADNI'
common_subs = set(os.listdir(path + '/amyloidPET_6mm')).intersection(set(os.listdir(path + '/tauPET_6mm')))
files = sorted(list(common_subs))

print("total subjects:", len(files))

mri_base = path + "/amyloidPET_6mm/{sub_id}/{timeline_t1}/T1.nii.gz"
amyloid_pet_base = path + "/amyloidPET_6mm/{sub_id}/{timeline_t1}/PET_T1.nii.gz"
tau_pet_base = path + "/tauPET_6mm/{sub_id}/{timeline_tau}/PET_T1.nii.gz"

num_sub = 0
subs = []
num_sub_multi_timeline = 0

mris = []
amyloid_pets = []
tau_pets = []

for file in files:
    assert os.path.isdir(os.path.join(path + '/amyloidPET_6mm', file)) and os.path.isdir(os.path.join(path + '/tauPET_6mm', file)), f"{file} does not have both amyloidPET and tauPET"
    
    timelines_t1 = os.listdir(os.path.join(path + '/amyloidPET_6mm', file))
    timelines_tau = os.listdir(os.path.join(path + '/tauPET_6mm', file))
    
    years_t1 = [timeline[:4] for timeline in timelines_t1]
    years_tau = [timeline[:4] for timeline in timelines_tau]

    # only keep the same year
    years = list(set(years_t1) & set(years_tau))
    if len(years) == 0:
        continue

    timelines_t1 = sorted([timeline for timeline in timelines_t1 if timeline[:4] in years])
    timelines_tau = sorted([timeline for timeline in timelines_tau if timeline[:4] in years])

    if len(timelines_t1) != len(timelines_tau):
        # remove the duplicate year
        years = set(years_t1) & set(years_tau)
        new_timelines_t1 = []
        for timeline in sorted(timelines_t1, reverse=True):
            year = timeline[:4]
            if year in years:
                years.remove(year)
                new_timelines_t1.append(timeline)
        timelines_t1 = new_timelines_t1

        years = set(years_t1) & set(years_tau)
        new_timelines_tau = []
        for timeline in sorted(timelines_tau, reverse=True):
            year = timeline[:4]
            if year in years:
                years.remove(year)
                new_timelines_tau.append(timeline)
        timelines_tau = new_timelines_tau

        assert len(timelines_t1) == len(timelines_tau), f"{file} has different number of timelines in amyloidPET and tauPET"

    appended = False
    for timeline_t1, timeline_tau in zip(timelines_t1, timelines_tau):
        assert timeline_t1[:4] == timeline_tau[:4], f"{file} has different year in amyloidPET and tauPET"

        mri = mri_base.format(sub_id=file, timeline_t1=timeline_t1)
        amyloid_pet = amyloid_pet_base.format(sub_id=file, timeline_t1=timeline_t1)
        tau_pet = tau_pet_base.format(sub_id=file, timeline_tau=timeline_tau)

        if not os.path.exists(mri) or not os.path.exists(amyloid_pet) or not os.path.exists(tau_pet):
            continue
        mris.append(mri)
        amyloid_pets.append(amyloid_pet)
        tau_pets.append(tau_pet)
        appended = True

    if appended:
        num_sub += 1
        subs.append(file)

        years = list(set(years_t1) & set(years_tau))
        if len(years) > 1:
            num_sub_multi_timeline += 1

print("subjects:", num_sub, "/", len(files))
print("unique subjects:", len(set(subs)))
print("subjects with multiple timelines:", num_sub_multi_timeline, "/", num_sub)
print("total data:", len(mris), len(amyloid_pets), len(tau_pets))

# %%
import json
import numpy as np
import pandas as pd

df = pd.read_csv('/scratch/faculty/shi/spectrum/jiaxiny/Datasets/ADNI/information/All_Subjects_CDR_13Nov2024.csv')
df = df[df['PTID'].isin(subs)]

data = []
for mri, amyloid_pet, tau_pet in zip(mris, amyloid_pets, tau_pets):
    # get CDR
    sub_id = mri.split('/')[-3]
    timeline = mri.split('/')[-2]
    timeline = timeline[:4] + '-' + timeline[4:6] + '-' + timeline[6:]
    df_sub = df[df['PTID'] == sub_id]
    # check if within 6 months
    df_sub = df_sub[df_sub['VISDATE'].apply(lambda x: abs(pd.to_datetime(x) - pd.to_datetime(timeline)).days < 180 if pd.notnull(x) else False)]
    try:
        assert len(df_sub) > 0, f"{mri} has no corresponding information in the csv file"
    except:
        continue
    df_sub = df_sub[df_sub['CDGLOBAL'].notnull()]
    if len(df_sub) == 0:
        continue
    if len(df_sub) > 1:
        # choose the one with the closest date
        df_sub = df_sub.iloc[np.argmin(df_sub['VISDATE'].apply(lambda x: abs(pd.to_datetime(x) - pd.to_datetime(timeline)).days))]
    else:
        df_sub = df_sub.iloc[0]
    assert type(df_sub['CDGLOBAL']) == np.float64, f"{mri} has non-float CDR value"
    cdr = df_sub['CDGLOBAL']

    year_mri = mri.split('/')[-2][:4]
    year_pet = amyloid_pet.split('/')[-2][:4]
    assert year_mri == year_pet, f"{mri} and {amyloid_pet} have different years"
    data.append({"mri": mri, "amyloid_pet": amyloid_pet, "tau_pet": tau_pet, "cdr": cdr})

print("total data:", len(data))

with open('/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MICCAI2024_segmentation/ALBEF/data/ADNI.json', 'w') as f:
    json.dump(data, f)

# %%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

data = nib.load('/ifs/loni/faculty/shi/spectrum/jiaxiny/HABLE/images/4106/T1/20230515/I1701456/Freesurfer6/4106/pet_uniform/amyloid_smoothed_inferior_cerebellum/T1.nii.gz').get_fdata().astype(np.float32)

# %%
plt.imshow(data[:, 160, :])