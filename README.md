# Saliency Analysis for Cardiac MR Motion Artefact Detection
This repository aims to provide the necessary implementations and guidelines for performing an analysis of motion artefacts within images through a variety of saliency detectors.

- [Preparing Environment](#preparing-environment)
- [Training and Evaluation](#training-and-evaluation)
- [Hyperparameters](#hyperparameters)
- [Results](#results)

# Preparing Environment

Before proceeding with the guideline, it is necessary to install the required Python packages by typing 
`pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html` and installing PyTorch v1.5.1 with GPU support. This code is tested under Python 3.7 Miniconda. 

In order to run the codeblocks of `saliency_visualization.ipynb` file, it is necessary to download a pre-trained model file and install a couple packages with `sh requirements_visualize.sh` command. It may also be necessary to install `node.js` and run `jupyter labextension install @jupyter-widgets/jupyterlab-manager` in order to use the interact panel of the `ipywidget` library on Jupyter Lab. 

ACDC Dataset is required as a prerequisite for reproducing these experiments. Although it is originally presented as CMR patient records in terms of NifTi files, patient records will be stored as `.npy` files which contain individual image slices. We will also generate corrupted image slices and split the data based on the original class IDs provided in the dataset. For example, the dataset file hierarchy will have such a form for a single corruption ratio of 0.15:

```
structured_dataset
├── cor_samp_test
│   └── 0.15
|   │   └── patient019_sigma1.0_z0_fn0.npy
|   │   └── patient019_sigma1.0_z0_fn1.npy
            .
            .
|   │   └── patient100_sigma1.0_z7_fn33.npy
├── cor_samp_train
│   └── 0.15
|   │   └── patient001_sigma1.0_z0_fn0.npy
|   │   └── patient001_sigma1.0_z0_fn1.npy
            .
            .
|   │   └── patient094_sigma1.0_z9_fn13.npy
├── cor_samp_val
│   └── 0.15
|   │   └── patient015_sigma1.0_z0_fn0.npy
|   │   └── patient015_sigma1.0_z0_fn1.npy
│   │       .
│   │       .
|   │   └── patient098_sigma1.0_z6_fn21.npy
├── uncor_samp_test
│   └── 0.15
|   │   └── patient019_z0_fn0.npy
|   │   └── patient019_z0_fn1.npy
            .
            .
|   │   └── patient100_z7_fn33.npy
├── uncor_samp_train
│   └── 0.15
|   │   └── patient001_z0_fn0.npy
|   │   └── patient001_z0_fn1.npy
            .
            .
|   │   └── patient094_z9_fn13.npy
├── uncor_samp_val
│   └── 0.15
|   │   └── patient015_z0_fn0.npy
|   │   └── patient015_z0_fn1.npy
│   │       .
│   │       .
|   │   └── patient098_z6_fn21.npy
```

In order to prepare such a folder structure, one should fill 2 YAML configuration files, namely, `construct_data_example.yaml` to generate the motion artefacts and image slices, and `split_data_example.yaml` for splitting the data into the training, validation and testing sets. After that, the following commands are sufficient for preparation of the dataset:
```
python generate_data.py --yaml_path=config/generate_data_example.yaml
python split_data.py --yaml_path=config/split_data_example.yaml
```

# Training and Evaluation
To train a model for detecting motion artefacts, another YAML configuration file has to be created. We provide a template file named `example_train_test_config.yaml` in `config/` with necessary explainations. After creating a configuration file, one can train and evaluate using the configuration file by:
```
python train.py --yaml_path=config/example_train_test_config.yaml
python test.py --yaml_path=config/example_train_test_config.yaml
```

# Hyperparameters
The dataset was split into 70% for model training, 20% for model validation and 10% for testing the accuracy of the prediction model in terms of their patient identification number. Also, 7 different corruption ratios were used for the generation of low-quality cMRI image samples: `C = {0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.40}`. The sigma parameter that will be used to sample the value from the Gaussian distribution has been taken as 1.0. 

# Results

## Accuracies on the testing split of the ACDC Dataset for motion artefact detection for different models
| Model / C  | 0.01   | 0.03   | 0.05   | 0.10   | 0.15   | 0.20   | 0.40   |
|------------|--------|--------|--------|--------|--------|--------|--------|
| ResNet-18  | 0.6824 | 0.7967 | 0.8500 | 0.9213 | 0.9485 | 0.9524 | 0.9987 |
| ResNet-34  | 0.6734 | 0.7884 | 0.8404 | 0.9090 | 0.9595 | 0.9499 | 0.9635 |
| ResNet-50  | 0.7381 | 0.8508 | 0.9034 | 0.9508 | 0.9800 | 0.9777 | 0.9966 |
| ResNet-101 | 0.7220 | 0.8531 | 0.9370 | 0.9320 | 0.9593 | 0.9507 | 0.9894 |

## F-1 Scores on the Interclass testing splits of the ACDC Dataset for motion artefact detection
| Train \ Test | 0.01 | 0.03 | 0.05 | 0.10 | 0.15 | 0.20 | 0.40 | 
|------------|--------|--------|--------|--------|--------|--------|--------|
| 0.01 | 0.689 | 0.705 | 0.709 | 0.709 | 0.709 | 0.709 | 0.709 |
| 0.03 | 0.680 | 0.761 | 0.775 | 0.781 | 0.781 | 0.781 | 0.781 |
| 0.05 | 0.711 | 0.792 | 0.802 | 0.809 | 0.809 | 0.809 | 0.809 |
| 0.10 | 0.652 | 0.803 | 0.880 | 0.901 | 0.905 |0.905 |0.905 |
| 0.15 | 0.666 | 0.817 | 0.919 | 0.975 | 0.977 | 0.978 | 0.978 |
| 0.20 | 0.658 | 0.776 | 0.880 | 0.970 | 0.971 | 0.972 | 0.972 |
| 0.40 | 0.665 | 0.695 | 0.759 | 0.923 | 0.976 | 0.990 | 0.993 |

# Citation

If you use code for your research, please cite the corresponding papers.

```
@inproceedings{Oksuz2018a
    title = {Cardiac {MR} Motion Artefact Correction from K-space Using Deep Learning-Based Reconstruction},
    author = {Ilkay {\"{O}}ks{\"{u}}z and James R. Clough and Aur{\'{e}}lien Bustin and Gastao Cruz and Claudia Prieto and Ren{\'{e}} M. Botnar and Daniel Rueckert and Julia A. Schnabel and Andrew P. King},
    booktitle = {Machine Learning for Medical Image Reconstruction - First International Workshop, {MLMIR} 2018, Held in Conjunction with {MICCAI} 2018}
    year = {2018}
}

@inproceedings{Oksuz2018b
    title = {Deep Learning Using K-Space Based Data Augmentation for Automated Cardiac {MR} Motion Artefact Detection},
    author = {Ilkay {\"{O}}ks{\"{u}}z and Bram Ruijsink and Esther Puyol{-}Ant{\'{o}}n and Aur{\'{e}}lien Bustin and Gastao Cruz and Claudia Prieto and Daniel Rueckert and Julia A. Schnabel and Andrew P. King},
    booktitle = {Medical Image Computing and Computer Assisted Intervention - {MICCAI} 2018 - 21st International Conference}
    year = {2018}
}
```
