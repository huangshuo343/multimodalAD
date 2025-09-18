# %%
import os
import json
import numpy as np

json_path = '/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MICCAI2024_segmentation/ALBEF/data/ADNI.json'
with open(json_path, 'r') as f:
    data = json.load(f)

train = data[:int(len(data)*0.8)]
# for each cdr = 0.5, duplicate 1 time
train += [per for per in train if per['cdr'] == 0.5]
# for each cdr >= 1, duplicate 5 times
train += [per for per in train if per['cdr'] >= 1.] * 5
val = data[int(len(data)*0.8):int(len(data)*0.9)]
test = data[int(len(data)*0.9):]

train_subs = set([per['mri'].split('/')[-3] for per in train])
val_subs = set([per['mri'].split('/')[-3] for per in val])
test_subs = set([per['mri'].split('/')[-3] for per in test])

# make sure no overlap
assert len(train_subs.intersection(val_subs)) == 0
assert len(train_subs.intersection(test_subs)) == 0
assert len(val_subs.intersection(test_subs)) == 0

print(len(train), len(val), len(test))

train_cdr = {0.: 0, 0.5: 0, 1.: 0}
val_cdr = {0.: 0, 0.5: 0, 1.: 0}
test_cdr = {0.: 0, 0.5: 0, 1.: 0}

for per in train:
    cdr = float(per['cdr'])
    cdr = cdr if cdr < 1. else 1.
    train_cdr[cdr] += 1

for per in val:
    cdr = float(per['cdr'])
    cdr = cdr if cdr < 1. else 1.
    val_cdr[cdr] += 1

for per in test:
    cdr = float(per['cdr'])
    cdr = cdr if cdr < 1. else 1.
    test_cdr[cdr] += 1

print(train_cdr)
print(val_cdr)
print(test_cdr)

# save to json
train_path = '/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MICCAI2024_segmentation/ALBEF/data/ADNI_train.json'
val_path = '/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MICCAI2024_segmentation/ALBEF/data/ADNI_val.json'
test_path = '/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MICCAI2024_segmentation/ALBEF/data/ADNI_test.json'

with open(train_path, 'w') as f:
    json.dump(train, f)

with open(val_path, 'w') as f:
    json.dump(val, f)

with open(test_path, 'w') as f:
    json.dump(test, f)