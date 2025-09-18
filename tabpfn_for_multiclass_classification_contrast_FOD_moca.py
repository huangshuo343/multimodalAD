# %%
#  Copyright (c) Prior Labs GmbH 2025.
import sys
sys.path.append('/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/TabPFN/src')

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import numpy as np
from tabpfn import TabPFNClassifier

# Load data
# X, y = load_iris(return_X_y=True)

csv_file = '/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/scripts/data_list_MOCAnocopy.csv'
import pandas as pd
df = pd.read_csv(csv_file)
# df = df.iloc[1441 : , : ]

train_data, temp_data = train_test_split(df, test_size=(1 - 0.6), random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
# data_number = 0
# if dataset == 'train':
# Make a copy of train_data to modify
data_train = train_data.copy()

# List to store augmented data
extra_rows = []

print("Training class distribution:\n", data_train['Diagnosis'].value_counts(normalize=True))

for _, one_data in train_data.iterrows():  # Use iterrows() for row-wise access
    if one_data['Diagnosis'] == 2:
        extra_rows.append(one_data)
    elif one_data['Diagnosis'] == 3:
        extra_rows.extend([one_data] * 3)  # Append the row 3 times

# Convert list to DataFrame and concatenate
if extra_rows:
    extra_df = pd.DataFrame(extra_rows)
    data_train = pd.concat([data_train, extra_df], ignore_index=True)

print("Expanded training class distribution:\n", data_train['Diagnosis'].value_counts(normalize=True))
# elif dataset == 'val':
data_val = validation_data.copy()
# data_dataframe = pd.DataFrame(self.data)
print("Validation class distribution:\n", data_val['Diagnosis'].value_counts(normalize=True))
# elif dataset == 'test':
data_test = test_data.copy()
# data_dataframe = pd.DataFrame(self.data)
print("Test class distribution:\n", data_test['Diagnosis'].value_counts(normalize=True))

# print(df.shape)
print(data_train.shape)
print(data_val.shape)
print(data_test.shape)
print('Training diagnosis for MOCA data are: ', data_train['Diagnosis'])

feature_path = '/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/ALBEF/output/test_pretrain_2025-02-07_18-37-36'
# load img features
# tr_features = np.load(feature_path + '/tr_feat_aligned.npz')
# tr_features = tr_features['aligned']
# val_features = np.load(feature_path + '/val_feat_aligned.npz')
# val_features = val_features['aligned']
# test_features = np.load(feature_path + '/test_feat_aligned.npz')
# test_features = test_features['aligned']
tr_features = np.load(feature_path + '/tr_feat_fusioned.npz')
tr_features = tr_features['fusioned']
val_features = np.load(feature_path + '/val_feat_fusioned.npz')
val_features = val_features['fusioned']
test_features = np.load(feature_path + '/test_feat_fusioned.npz')
test_features = test_features['fusioned']
tr_diagnosis = np.load(feature_path + '/tr_label.npz')
tr_diagnosis = tr_diagnosis['label']
val_diagnosis = np.load(feature_path + '/val_label.npz')
val_diagnosis = val_diagnosis['label']
test_diagnosis = np.load(feature_path + '/test_label.npz')
test_diagnosis = test_diagnosis['label']
# tr_features = tr_features[0 : 848, : ]
# print(tr_features[ : , -1])
# print(val_features[ : , -1])
# print(test_features[ : , -1])
# tr_features_downsample = tr_features[ : , : : 16]
# val_features_downsample = val_features[ : , : : 16]
# test_features_downsample = test_features[ : , : : 16]
# for i in range(15):
#     # tr_features_downsample = tr_features_downsample + tr_features[ : , i + 1 : : 16]
#     # val_features_downsample = val_features_downsample + val_features[ : , i + 1 : : 16]
#     # test_features_downsample = test_features_downsample + test_features[ : , i + 1 : : 16]
#     tr_features_downsample = np.where(np.abs(tr_features_downsample) > np.abs(tr_features[ : , i + 1 : : 16]), tr_features_downsample, tr_features[ : , i + 1 : : 16])
#     val_features_downsample = np.where(np.abs(val_features_downsample) > np.abs(val_features[ : , i + 1 : : 16]), val_features_downsample, val_features[ : , i + 1 : : 16])
#     test_features_downsample = np.where(np.abs(test_features_downsample) > np.abs(test_features[ : , i + 1 : : 16]), test_features_downsample, test_features[ : , i + 1 : : 16])
# tr_features = tr_features_downsample.copy()
# val_features = val_features_downsample.copy()
# test_features = test_features_downsample.copy()
# tr_features = tr_features[0 : 850, : ]

# print("Available keys:", tr_diagnosis.files)  
# print(tr_features['aligned'].shape)

# img_diagnosis = np.concatenate([tr_features[ : , -1], val_features[ : , -1], test_features[ : , -1]], axis=0)
test_diagnosis = np.concatenate([val_diagnosis, test_diagnosis], axis=0)

# tr_diagnosis = np.where(tr_diagnosis > 1, 1, 0)
# test_diagnosis = np.where(test_diagnosis > 1, 1, 0)

# tr_features = np.mean(tr_features[ : , : -1], axis=1)
# tr_features = tr_features / np.max(tr_features)
# val_features = np.mean(val_features[ : , : -1], axis=1)
# val_features = val_features / np.max(val_features)
# test_features = np.mean(test_features[ : , : -1], axis=1)
# test_features = test_features / np.max(test_features)

# img_features = np.concatenate([tr_features[ : , : -1], val_features[ : , : -1], test_features[ : , : -1]], axis=0)
test_features = np.concatenate([val_features, test_features], axis=0)

# img_features = np.random.permutation(img_features)
# img_features = np.random.rand(*img_features.shape)
# img_features = np.zeros_like(img_features)

# img_features = img_features[ : , 200 : ]

# feature_path = '/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/ALBEF/output/test_pretrain_2025-02-07_18-37-36'
# load img features
FOD_trainfeatures = np.load('/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/swinad/train_features2_random.npy')
FOD_valfeatures = np.load('/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/swinad/val_features2_random.npy')
FOD_testfeatures = np.load('/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/swinad/test_features2_random.npy')
# tr_features = tr_features[0 : 848, : ]
print(FOD_trainfeatures[ : , -1])
print(FOD_valfeatures[ : , -1])
print(FOD_testfeatures[ : , -1])
print(FOD_trainfeatures.shape)
# Make a copy of train_data to modify
features_train = FOD_trainfeatures.copy()
features_train_diagnosis = FOD_trainfeatures.copy()
features_train = (FOD_trainfeatures - np.min(FOD_trainfeatures, axis=0)) / (np.max(FOD_trainfeatures, axis=0) - np.min(FOD_trainfeatures, axis=0))
FOD_valfeatures = (FOD_valfeatures - np.min(FOD_trainfeatures, axis=0)) / (np.max(FOD_trainfeatures, axis=0) - np.min(FOD_trainfeatures, axis=0))
FOD_testfeatures = (FOD_testfeatures - np.min(FOD_trainfeatures, axis=0)) / (np.max(FOD_trainfeatures, axis=0) - np.min(FOD_trainfeatures, axis=0))

features_train[ : , -1] = features_train_diagnosis[ : , -1]
print("Training class distribution:\n", dict(zip(*np.unique(features_train[:, -1], return_counts=True))))

# List to store augmented data
extra_rows = []

for one_data in features_train:  
    if one_data[-1] == 1:
        extra_rows.append(one_data.copy())  # Copy to avoid modifying the original reference
    elif one_data[-1] == 2:
        extra_rows.extend([one_data.copy()] * 3)  # Duplicate row 3 times

# Convert extra_rows list to NumPy array
extra_rows = np.array(extra_rows)

# Concatenate with original NumPy array
features_train = np.vstack([features_train, extra_rows])

# Print new class distribution
unique, counts = np.unique(features_train[:, -1], return_counts=True)
print("Updated training class distribution:", dict(zip(unique, counts)))
FOD_trainfeatures = features_train.copy()
print(FOD_trainfeatures.shape)
# print(FOD_trainfeatures[ : , -1])

FOD_features_test = np.concatenate([FOD_valfeatures, FOD_testfeatures], axis=0)
# FOD_features_test = np.random.rand(*FOD_features_test.shape)
random_matrix = np.random.randn(*FOD_features_test.shape)
random_select = np.random.rand(*FOD_features_test.shape)
# select_percentage = 0.1
# FOD_features_test = FOD_features_test * select_percentage + random_matrix * (1 - select_percentage)
# FOD_features_test = np.where(random_select > 1 - select_percentage, FOD_features_test, random_matrix)

# print(tr_features.shape)
# print(val_features.shape)
# print(test_features.shape)
# print('train features are: ', tr_features)
print('train features shape: ', FOD_trainfeatures.shape)
print('test features shape: ', FOD_features_test.shape)
print('train diagnosis are: ', tr_diagnosis)

# combine img features with df in column
# df = pd.concat([df, pd.DataFrame(img_features[:, :-1])], axis=1)
# df = pd.concat(
#     [df.reset_index(drop=True), pd.DataFrame(img_features).reset_index(drop=True)], 
#     axis=1
# )

# data_train = data_train.drop(columns=['Diagnosis'])
tr_features = pd.concat(
    [data_train.reset_index(drop=True), pd.DataFrame(tr_features).reset_index(drop=True)], 
    axis=1
)
print('train features shape: ', tr_features.shape)
# tr_features = pd.concat(
#     [tr_features.reset_index(drop=True), pd.DataFrame(FOD_trainfeatures[ : , : 64]).reset_index(drop=True)], 
#     axis=1
# )
# tr_features = data_train

data_test = pd.concat(
    [data_val.reset_index(drop=True), data_test.reset_index(drop=True)],
    axis=0
)
# data_test = data_test.drop(columns=['Diagnosis'])
test_features = pd.concat(
    [data_test.reset_index(drop=True), pd.DataFrame(test_features).reset_index(drop=True)],
    axis=1
)
print('test features shape: ', test_features.shape)
# test_features = pd.concat(
#     [test_features.reset_index(drop=True), pd.DataFrame(FOD_features_test[ : , : 64]).reset_index(drop=True)],
#     axis=1
# )
#print('test features shape: ', FOD_features_test.shape)
print('test features shape: ', test_features.shape)
# test_features = data_test

# data_val = data_val.drop(columns=['Diagnosis'])

# df = pd.concat(
#     [df.reset_index(drop=True), pd.DataFrame(img_features).reset_index(drop=True)], 
#     axis=1
# )

# print('dataframe shape is: ' + str(df.shape))
# df = df[np.sum(df.isna().to_numpy(), axis=1) <= 5]
# # fill na with nearest value
# df = df.ffill()


# print('dataframe shape is: ' + str(df.shape))
# # print(df.columns)

# df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
# df = df.select_dtypes(include=['number'])

print('dataframe shape is: ' + str(tr_features.shape))
tr_features_all = tr_features.copy()
tr_features = tr_features[np.sum(tr_features.isna().to_numpy(), axis=1) <= 5]
# fill na with nearest value
tr_features = tr_features.ffill()

FOD_data_dataframe = pd.DataFrame(FOD_trainfeatures[ : , : 3])
FOD_data_dataframe = FOD_data_dataframe[np.sum(tr_features_all.isna().to_numpy(), axis=1) <= 5]

tr_features = pd.concat(
    [tr_features.reset_index(drop=True), FOD_data_dataframe.reset_index(drop=True)], 
    axis=1
)


print('dataframe shape is: ' + str(tr_features.shape))
# print(df.columns)

tr_features['Sex'] = tr_features['Sex'].map({'M': 0, 'F': 1})
tr_features = tr_features.select_dtypes(include=['number'])

# print(tr_features.columns)
tr_diagnosis = tr_features['Diagnosis'].to_numpy().ravel()
tr_features = tr_features.drop(columns=['Diagnosis']).to_numpy()
# tr_features = FOD_data_dataframe.to_numpy()

print('dataframe shape is: ' + str(test_features.shape))
test_features_all = test_features.copy()
test_features = test_features[np.sum(test_features.isna().to_numpy(), axis=1) <= 5]
# fill na with nearest value
test_features = test_features.ffill()

FOD_data_dataframe = pd.DataFrame(FOD_features_test[ : , : 3])
FOD_data_dataframe = FOD_data_dataframe[np.sum(test_features_all.isna().to_numpy(), axis=1) <= 5]

test_features = pd.concat(
    [test_features.reset_index(drop=True), FOD_data_dataframe.reset_index(drop=True)], 
    axis=1
)


print('dataframe shape is: ' + str(test_features.shape))
# print(df.columns)

# print(test_features['Sex'])
test_features['Sex'] = test_features['Sex'].map({'M': 0, 'F': 1})
test_features = test_features.select_dtypes(include=['number'])

test_diagnosis = test_features['Diagnosis'].to_numpy().ravel()
test_features = test_features.drop(columns=['Diagnosis']).to_numpy()
# test_features = FOD_data_dataframe.to_numpy()

# df = df.dropna()

# X = df.drop(columns=['Diagnosis']).to_numpy()
# X = tr_features
# X = img_features# [ : , 0 : -1]
# X = X[0 : 1441, : ]

# y = df[['Diagnosis']].to_numpy().ravel()
# y = img_diagnosis
# y = y[0 : 1441]


# # read csv file
# taupet = '/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MICCAI2025_multimodal/llm/ADNI_phenotypes.csv'
# df = pd.read_csv(taupet)

# # PTID	PTAGE	PTGENDER	PTEDUCAT	positivity	DIAGNOSIS	frontal_suvr	parietal_suvr	occipital_suvr	temporal_suvr	other_ctx_suvr
# df = df.drop(columns=['PTID', 'positivity'])
# df['PTGENDER'] = df['PTGENDER'].map({'Male': 1, 'Female': 2})
# df = df.dropna()

# X = df.drop(columns=['DIAGNOSIS']).to_numpy()

# y = df[['DIAGNOSIS']].to_numpy().ravel()
# # y[y >= 2] = 2
# # y -= 1
# %%
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.33,
#     random_state=42,
# )

# X_train, X_test, y_train, y_test = train_test_split(
#     tr_features,
#     tr_diagnosis,
#     test_size=0.33,
#     random_state=42,
# )

# X_train = X[ : 848] # 150
# X_test = X[848 : ] # 150
# y_train = y[ : 848] # 150
# y_test = y[848 : ] # 150

# X_train = X[848 : ] # 150
# X_test = X[ : 848] # 150
# y_train = y[848 : ] # 150
# y_test = y[ : 848] # 150
X_train = tr_features # [ : , 39 : ] #[ : 688, : ]
X_test = test_features # [ : , 39 : ]
y_train = tr_diagnosis # [ : 688]
y_test = test_diagnosis

# X_train = X[848 : 998] # 150
# X_test = X[998 : ] # 150
# y_train = y[848 : 998] # 150
# y_test = y[998 : ] # 150

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialize a classifier
TabPFNclf = TabPFNClassifier(ignore_pretraining_limits=True)
TabPFNclf.fit(X_train, y_train)

# Predict labels
predictions = TabPFNclf.predict(X_test)

print("Accuracy", accuracy_score(y_test, predictions))


# tp = (y_test == 1) & (predictions == 1)
# tn = (y_test == 0) & (predictions == 0)
# fp = (y_test == 0) & (predictions == 1)
# fn = (y_test == 1) & (predictions == 0)

# accuracy = (tp.sum() + tn.sum()) / len(y_test)
# sensitivity = tp.sum() / (tp.sum() + fn.sum())
# specificity = tn.sum() / (tn.sum() + fp.sum())
# f1 = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
# print("Accuracy", accuracy)
# print("Sensitivity", sensitivity)
# print("Specificity", specificity)
# print("F1", f1)



# confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predictions))


# Predict probabilities
prediction_probabilities = TabPFNclf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities, multi_class="ovr"))

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score

# Confusion matrix
# conf_matrix = np.array([[209,  34,   0],
#                         [ 49,  87,   7],
#                         [  2,  28,  32]])
conf_matrix = confusion_matrix(y_test, predictions)

# Extracting true labels and predicted labels
y_true = y_test # np.repeat(np.arange(3), conf_matrix.sum(axis=1))
y_pred = predictions # np.concatenate([np.repeat(i, conf_matrix[i, :].sum()) for i in range(3)])

# Compute Precision, Recall, F1-score per class
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

# Compute Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_true, y_pred)

# Compute Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_true, y_pred)

# Print results
print("Precision (per class):", precision)
print("Recall (per class):", recall)
print("F1-score (per class):", f1)
print("Balanced Accuracy:", balanced_acc)
print("Matthews Correlation Coefficient (MCC):", mcc)

# svm classifier
from sklearn.svm import SVC

"""
Accuracy 0.7727272727272727
[[107   2   1]
 [ 16   6   6]
 [  3  12  23]]
"""

# standardize data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    clf = SVC(kernel=kernel)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    print(f"SVC Accuracy ({kernel})", accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
# %%
import shap
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Optional: initialize JS visualization (works in Jupyter notebooks)
shap.initjs()

TabPFNclf = TabPFNClassifier(ignore_pretraining_limits=True)
TabPFNclf.fit(X_train[..., : 100], y_train[...]) # 10

background = shap.sample(X_train[..., : 100], 500, random_state=42) # 10 100

explainer = shap.KernelExplainer(TabPFNclf.predict_proba, background)

shap_values = explainer.shap_values(X_test[..., : 100], nsamples=500) # 10 100 1000

# explainer = shap.GradientExplainer(TabPFNclf.predict_proba, background)
# shap_values = explainer.shap_values(X_test[..., : ])

np.save('/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/TabPFN/runs' + '/' + 'features_importances_valuestest.npy', shap_values)
np.save('/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/TabPFN/runs' + '/' + 'features_importances_test.npy', X_test[..., : ])
np.save('/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/multimodel/SwinUNETR/TabPFN/runs' + '/' + 'features_importances_train.npy', X_train[..., : ])

# %%
for i in range(shap_values.shape[-1]):
    class_shap_values = shap_values[...,i]
    print(f"SHAP summary for class {i}:")
    shap.summary_plot(class_shap_values, X_test[..., : ]) # 10 
