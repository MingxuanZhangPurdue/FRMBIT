import pandas as pd
import numpy as np
from functools import reduce


train_id_filename = "datasets/autocomplete_brain_heart_train_May1_2024.id.list"
with open(train_id_filename) as file:
    train_ids = [int(line.rstrip()) for line in file]
train_ids = np.array(train_ids)

test_id_filename = "datasets/autocomplete_brain_heart_test_May1_2024.id.list"
with open(test_id_filename) as file:
    test_ids = [int(line.rstrip()) for line in file]
test_ids = np.array(test_ids)

heart = pd.read_csv("datasets/UKB_heart_bai82_Jan21_2021_QC_5mad_scale_resid_Sep19_2023.csv")
abd = pd.read_csv("datasets/ukb_abdominal_idp_41_042923_all_5mad_scale_resid_Sep19_2023.csv")
tfmri = pd.read_csv("datasets/tfMRI_2_all_5mad_scale_resid_Jun19_2023.csv")
smri = pd.read_csv("datasets/sMRI_2_all_5mad_scale_resid_Jun19_2023.csv")
rfmri = pd.read_csv("datasets/rfMRI_2_all_5mad_scale_resid_Jun19_2023.csv")
dmri = pd.read_csv("datasets/dMRI_2_all_5mad_scale_resid_Jun19_2023.csv")

dfs = [heart, abd, tfmri, smri, rfmri, dmri]
merged_df = reduce(lambda left, right: pd.merge(left, right, on=['FID', 'IID'], how='outer'), dfs)
merged_df = merged_df.drop("FID", axis=1)
merged_df = merged_df.rename(columns={'IID': 'ID'})
train = merged_df[merged_df['ID'].isin(train_ids)]
test  = merged_df[merged_df['ID'].isin(test_ids)]

train.to_csv("datasets/train.csv", index=False)
test.to_csv("datasets/test.csv", index=False)