import argparse
import yaml
import tqdm

import torchvision
from torchvision import transforms, models, utils, datasets

import torch
from torch.utils.data import Dataset, DataLoader
from dataset import ACDCDataset, Resize, ToTensor, Normalize,\
                    OneToThreeDimension
from models.models import get_model
from utils.utils import get_most_recent_model, yaml_var_concat


def test(model, test_set, dataset_size, device=None):
    model.eval()

    running_corrects = 0.
    running_tp = 0
    running_tn = 0
    running_fp = 0
    running_fn = 0

    for inputs, labels in tqdm.tqdm(test_set):
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

        running_tp += torch.sum(preds * labels == 1).float()
        running_tn += torch.sum(preds * labels == 0).float()
        false_samps = preds * labels == 0
        running_fp += torch.sum(false_samps == (preds == 1)).float()
        running_fn += torch.sum(false_samps == (labels == 1)).float()

    acc = running_corrects / dataset_size
    epoch_prec = running_tp / (running_tp + running_fp)
    epoch_rec = running_tp / (running_tp + running_fn)

    print("Test Acc: {:.6f}".format(acc))
    print("Test Precision: {:.6f} Test Recall: {:6f}".format(
           epoch_prec, epoch_rec))

def main():
    parser = argparse.ArgumentParser(description='Binary MRI Quality Classification')
    parser.add_argument('--yaml_path', type=str, metavar='YAML',
                        default="config/acdc_binary_classification.yaml",
                        help='Enter the path for the YAML config')
    args = parser.parse_args()

    yaml.add_constructor("!join", yaml_var_concat)

    yaml_path = args.yaml_path
    with open(yaml_path, 'r') as f:
        train_args = yaml.load(f, Loader=yaml.Loader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    composed = transforms.Compose([Resize((224, 224)),
                                   OneToThreeDimension(),
                                   ToTensor(),
                                   Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                                  ])

    acdc_dataset = ACDCDataset(train_args["pos_samps_test"],
                               train_args["neg_samps_test"],
                               transform=composed)

    dataloader = DataLoader(acdc_dataset,
                            batch_size=train_args["batch_size"],
                            shuffle=False, num_workers=4)
    dataset_size = len(acdc_dataset)

    model_ft = get_model(train_args["model"], device,
                         pretrained=train_args["pretrained"])
    state = get_most_recent_model(train_args["model"],
                                  train_args["model_save_dir"])
    model_ft.load_state_dict(state)

    test(model_ft, dataloader, dataset_size, device=device)
    

if __name__ == "__main__":
    main()
