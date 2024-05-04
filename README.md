# Learning a foundational representation of UK Biobank brain imaging traits

All the commnad lines provided below should be executed from the folder that contains this README.md file.

## Set up enviroment

```console
pip install requirements.txt
```

## Preprocessing the datasets

Please put all following data files inside the datasets folder

* autocomplete_brain_heart_train_May1_2024.id.list
* autocomplete_brain_heart_test_May1_2024.id.list
* UKB_heart_bai82_Jan21_2021_QC_5mad_scale_resid_Sep19_2023.csv
* ukb_abdominal_idp_41_042923_all_5mad_scale_resid_Sep19_2023.csv
* tfMRI_2_all_5mad_scale_resid_Jun19_2023.csv
* sMRI_2_all_5mad_scale_resid_Jun19_2023.csv
* rfMRI_2_all_5mad_scale_resid_Jun19_2023.csv
* dMRI_2_all_5mad_scale_resid_Jun19_2023.csv

Then run,

```console
python preprocessing.py
```

This will generate two additional CSV files: train.csv and test.csv inside the datasets folder.

## Train AutoComplete models

Below is an example training script, where we set learning rate: lr=0.1, batchsize: bs=512, copymask amount: cm=0.5, encoding ratio: er=0.5, model depth: depth=1, number of training epochs: epochs=100, and random seed: seed=42.

The trained model will be saved under the checkpoints folder, and the name for the saved model is specified by the **--save_model_path** argument.

```console
lr=1.0
bs=512
cm=0.5
er=0.5
depth=1
epochs=100
seed=42

python AutoComplete/fit.py \
    --data_file datasets/train.csv \
    --id_name ID \
    --copymask_amount ${cm} \
    --batch_size ${bs} \
    --epochs ${epochs} \
    --val_split 0.95 \
    --lr ${lr} \
    --device cuda \
    --encoding_ratio ${er} \
    --depth ${depth} \
    --seed ${seed} \
    --output checkpoints/seed${seed}_lr${lr}_bs${bs}_cm${cm}_er${er}_depth${depth}_epochs${epochs} \
    --save_model_path checkpoints/seed${seed}_lr${lr}_bs${bs}_cm${cm}_er${er}_depth${depth}_epochs${epochs}
```

## Inference trained AutoComplete models

For example, if we want to generate hidden representations for both the train and test CSV files using the saved model: **seed42_lr1.0_bs512_cm0.5_er0.5_depth1_epochs100.json**,


```console
python AutoComplete/inference.py \
    --impute_using_saved checkpoints/seed42_lr1.0_bs512_cm0.5_er0.5_depth1_epochs100 \
    --id_name ID \
    --device cuda
```


This will generate a new CSV file named as **encoding_ratio=0.5.csv** inside the results folder, which contains the hidden representations and IDs for all objects from the train and test sets.