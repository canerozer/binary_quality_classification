"""
This code is not working at the moment, there are two potential reasons.
1) The data is pretty indiscriminative in terms of the noise property. The std dev. of the noise could be increased.
2) Models with larger capacity can be tried. In some cases, the validation loss fells behind the training loss.
3) Currently there is no such a per-channel normalization scheme. Adding that might be obligatory.S
"""
import os
import copy
import argparse
import yaml
import tqdm
import time

import numpy as np
from skimage import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, models, utils, datasets
import nibabel as nib

from models.models import get_model
from dataset import ACDCDataset, Resize, ToTensor, Normalize,\
                    OneToThreeDimension

import matplotlib.pyplot as plt


def train(model, criterion, optimizer, scheduler=None, num_epochs=25,
          test_every=1, save_every=3):
    iter_epochs = range(num_epochs)

    if train_args["retrieve_last_model"]:
        prev_state = get_most_recent_state(train_args["model"],
                                           train_args["model_save_dir"])
        iter_epochs = range(prev_state["epoch"] + 1, num_epochs)
        optimizer.load_state_dict(prev_state["optimizer"])
        scheduler.load_state_dict(prev_state["scheduler"])
        model.state_dict(prev_state["model"])

    since = time.time()
    for epoch in iter_epochs:
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()

        running_loss_train = 0.0
        running_corrects_train = 0

        for inputs, labels in dataloader["train"]:
            inputs = inputs.to(device, dtype=torch.float32)
            #inputs = inputs.to(device, dtype=torch.double)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss_train += loss.item() * inputs.size(0)
            running_corrects_train += torch.sum(preds == labels.data)

        epoch_loss_train = running_loss_train / dataset_sizes["train"]
        epoch_acc_train = running_corrects_train.double() / dataset_sizes["train"]

        print('{} Loss: {:.6f} Acc: {:.6f}'.format(
               "Train", epoch_loss_train, epoch_acc_train))

        if epoch % test_every == 0:
            # optimizer.zero_grad()
            evaluate(model, criterion)

        if scheduler:
            scheduler.step()

        if epoch % save_every == 0: 
            if scheduler:
                state = {'epoch': epoch, 'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()}
            else:
                state = {'epoch': epoch, 'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()}

            torch.save(state,
                       train_args["model_save_dir"] + train_args["model"] +\
                       "_ep" + str(epoch) + ".pth")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


def evaluate(model, criterion):
    model.eval()

    running_loss = 0.0
    running_corrects = 0.

    for inputs, labels in dataloader["val"]:
        inputs = inputs.to(device, dtype=torch.float32)
        #inputs = inputs.to(device, dtype=torch.double)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes["val"]
    epoch_acc = running_corrects.double() / dataset_sizes["val"]

    print('{} Loss: {:.6f} Acc: {:.6f}'.format(
           "Val", epoch_loss, epoch_acc))


def test(model, test_set, dataset_size):
    model.eval()

    running_corrects = 0.

    for inputs, labels in test_set:
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects / dataset_size

    print("Test Acc: {:.6f}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binary MRI Quality Classification')
    parser.add_argument('--yaml_path', type=str, metavar='YAML',
                        default="config/acdc_binary_classification.yaml",
                        help='Enter the path for the YAML config')
    args = parser.parse_args()

    yaml_path = args.yaml_path
    with open(yaml_path, 'r') as f:
        train_args = yaml.safe_load(f)

    os.makedirs(train_args["model_save_dir"], exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    composed = transforms.Compose([Resize((224, 224)),
                                   OneToThreeDimension(),
                                   ToTensor(),
                                   Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                                  ])
    acdc_dataset = {x: ACDCDataset(train_args["pos_samps_"+x],
                                  train_args["neg_samps_"+x],
                                  transform=composed)
                   for x in ["train", "val", "test"]}

    dataloader = {x: DataLoader(acdc_dataset[x],
                                batch_size=train_args["batch_size"],
                                shuffle=True, num_workers=4,
                                # sampler=sampler[x]
                                )
                  for x in ["train", "val", "test"]}
    dataset_sizes = {x: len(acdc_dataset[x]) for x in ["train", "val", "test"]}

    model_ft = get_model(train_args["model"], device,
                         pretrained=train_args["pretrained"])

    criterion = nn.CrossEntropyLoss()

    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0003, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-5)

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    model_ft = train(model_ft, criterion, optimizer_ft, #exp_lr_scheduler,
                     num_epochs=train_args["epoch"])

    test(model_ft, dataloader["test"], dataset_sizes["test"])

