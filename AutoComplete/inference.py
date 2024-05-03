import pandas as pd
import numpy as np
import torch
import re
from torch.utils.data import DataLoader
from AutoComplete.dataset import CopymaskDataset
import argparse


parser = argparse.ArgumentParser(description='Inference trained AutoComplete model')
parser.add_argument('--impute_using_save', type=str, help='Path to saved model')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
id_name = "ID"
train_tab = pd.read_csv("datasets/train.csv").set_index(id_name)
test_tab = pd.read_csv("datasets/test.csv").set_index(id_name)
feature_dim = train_tab.shape[-1]

pattern = r"[-+]?\d*\.\d+|\d+"
nums_str = re.findall(pattern, args.impute_using_saved)
hyperparams = [float(num_str) for num_str in nums_str]
encoding_ratio = hyperparams[3]
copy_masking = hyperparams[2]
depth = int(hyperparams[4])
batch_size = int(hyperparams[1])


model = torch.load(args.impute_using_saved, map_location=torch.device(device))
model.eval()

ncats = train_tab.nunique()
binary_features = train_tab.columns[ncats == 2]
contin_features = train_tab.columns[~(ncats == 2)]
feature_ord = list(contin_features) + list(binary_features)
CONT_BINARY_SPLIT = len(contin_features)

train_dset = train_tab[feature_ord]
train_stats = dict(mean=train_dset.mean().values)
train_stats['std'] = np.nanstd(train_dset.values - train_stats['mean'], axis=0)

imptab = pd.concat([train_tab[feature_ord], test_tab[feature_ord]], axis=0)

print(f'(impute) Dataset size:', imptab.shape[0])

mat_imptab = (imptab.values - train_stats['mean'])/train_stats['std']

dset = DataLoader(
    CopymaskDataset(mat_imptab, 'final', copymask_amount=0.0),
    batch_size=batch_size,
    shuffle=False, num_workers=0)

hidden_ls = []
for bi, batch in enumerate(dset):
    datarow, _, masked_inds = batch
    datarow = datarow.float()

    with torch.no_grad():
        datarow = datarow.to(device)
        for i in range(depth):
            datarow = model.net[2*i](datarow)
            datarow = model.net[2*i+1](datarow)
    hidden_ls += [datarow.cpu().numpy()]

hmat = np.concatenate(hidden_ls)
df_hidden = pd.DataFrame(data=hmat)
df_hidden.insert(0, "ID", np.concatenate((train_tab.index, test_tab.index), axis=0), True)
df_hidden.to_csv("checkpoints/encoding_ratio="+str(encoding_ratio)+".csv", index=False)