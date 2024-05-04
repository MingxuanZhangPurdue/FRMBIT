# Learning a foundational representation of UK Biobank brain imaging traits

All the commnadlines provided below should be executed from the folder that contains this README.md file

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

This will generate two additional CSV files: train.csv and test.csv inside the datasets folder

