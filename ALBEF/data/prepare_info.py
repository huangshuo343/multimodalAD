# %%
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import csv
import json
import pandas as pd

base_path = '/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MICCAI2024_segmentation/ALBEF/data/hable_info'
csvs = [
    'HD 1 Non-Hispanic White 50+ Request 318.csv', # (1242, 1539)
    'HD 2 Non-Hispanic White 50+ Request 318.csv', # (624, 1475)
    'HD 3 Non-Hispanic White 50+ Request 318.csv', # (153, 1407)

    'HD 1 Mexican American 50+ Request 318.csv', # (1251, 1539)
    'HD 2 Mexican American 50+ Request 318.csv', # (553, 1475)
    'HD 3 Mexican American 50+ Request 318.csv', # (154, 1407)

    'HD 1 African American 50+ Request 318.csv', # (792, 1539)
]

csv2sub = {}
for csv_name in csvs:
    csv_path = os.path.join(base_path, csv_name)
    df = pd.read_csv(csv_path)
    print(df.shape)
    csv2sub[csv_name] = set(list(df['Med_ID'].values))

for i in range(len(csvs)):
    for j in range(i+1, len(csvs)):
        csv_name1 = csvs[i]
        csv_name2 = csvs[j]
        if csv_name1 == csv_name2:
            continue
        common = csv2sub[csv_name1] & csv2sub[csv_name2]
        if len(common) > 0:
            print(csv_name1, csv_name2)

# %%
json_path = '/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MICCAI2024_segmentation/ALBEF/data/multimodal.json'
with open(json_path, 'r') as f:
    data = json.load(f)

unique_sub = set()
for d in data:
    mri = d['mri']
    sub = mri.split('/')[-4]
    unique_sub.add(int(sub))

print(len(unique_sub))

sub2csv = {}
used_csv = set()
for sub in unique_sub:
    sub2csv[sub] = []
    for csv_name in csvs:
        if sub in csv2sub[csv_name]:
            sub2csv[sub].append(csv_name)
            used_csv.add(csv_name)
    if len(sub2csv[sub]) == 0:
        del sub2csv[sub]

print(len(sub2csv))
print(len(used_csv))