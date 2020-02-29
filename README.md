# Binary MRI Quality Classification
A repository for classifying cMRI images into two categories: high-quality and low-quality.

# Preparing Environment
ACDC cMRI Dataset is required as a prerequisite. The dataset is needed to be present in a folder and it will be used as the `src` parameter of the YAML configurations. Also, the required python packages can be installed by using the requirements.txt file by typing:
`pip install -r requirements.txt`

# Configuration
First of all, the YAML file regarding the low-quality data construction has to be configured. After that, the YAML file regarding the data splits (train/val/test) is needed to be filled. Running run.sh will automatically create the low-quality data using the high-quality ones, and it will sort the patients into train, validation and test, based on their patient identification number and the disease classification. Lastly, YAML files regarding the classification configuration can be filled and used for training a variety of models.

# Hyperparameters
The dataset was split into 70% for model training, 20% for model validation and 10% for testing the accuracy of the prediction model in terms of their patient identification number. Also, 7 different corruption ratios were used for the construction of low-quality cMRI image samples: `C = {0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.40}`. The sigma parameter that will be used to sample the value from the Gaussian distribution has been taken as 1.0. 

# Results
| Model / C  | 0.01   | 0.03   | 0.05   | 0.10   | 0.15   | 0.20   | 0.40   |
|------------|--------|--------|--------|--------|--------|--------|--------|
| ResNet-18  | 0.6824 | 0.7967 | 0.8500 | 0.9213 | 0.9485 | 0.9524 | 0.9987 |
| ResNet-34  | 0.6734 | 0.7884 | 0.8404 | 0.9090 | 0.9595 | 0.9499 | 0.9635 |
| ResNet-50  | 0.7381 | 0.8508 | 0.9034 | 0.9508 | 0.9800 | 0.9777 | 0.9966 |
| ResNet-101 | 0.7220 | 0.8531 | 0.9370 | 0.9320 | 0.9593 | 0.9507 | 0.9894 |

