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
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils
import nibabel as nib

from dataset import ACDCDataset, Rescale, RandomCrop, ToTensor

import matplotlib.pyplot as plt


def get_model(model_name, device):
    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    elif model_name == "resnet34":
        model_ft = models.resnet34(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    elif model_name == "vgg11":
        model_ft = models.vgg11(pretrained=False)
        model_ft.features[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        model_ft.classifier[6] = nn.Linear(4096, 2, bias=True)

    model_ft = model_ft.to(torch.double).to(device)
    return model_ft


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                #model.train()   # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train(model, criterion, optimizer, scheduler, num_epochs=25, test_every=1):
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes["train"]
        epoch_acc = running_corrects.double() / dataset_sizes["train"]

        print('{} Loss: {:.6f} Acc: {:.6f}'.format(
               "Train", epoch_loss, epoch_acc))

        if epoch != 0 and epoch % test_every == 0:
            optimizer.zero_grad()
            evaluate(model, criterion)

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


def evaluate(model, criterion):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader["val"]:
        inputs = inputs.to(device)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binary MRI Quality Classification')
    parser.add_argument('--yaml_path', type=str, metavar='YAML',
                        default="config/acdc_binary_classification.yaml",
                        help='Enter the path for the YAML config')
    args = parser.parse_args()

    yaml_path = args.yaml_path
    with open(yaml_path, 'r') as f:
        train_args = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    composed = transforms.Compose([#Rescale(256),
                                   RandomCrop(224),
                                   ToTensor(),
                                  ])
    acdc_dataset = {x: ACDCDataset(train_args["pos_samps_"+x],
                                   train_args["neg_samps_"+x],
                                   transform=composed)
                    for x in ["train", "val"]}

    dataloader = {x: DataLoader(acdc_dataset[x], batch_size=64,
                                shuffle=False, num_workers=4)
                  for x in ["train", "val"]}

    dataset_sizes = {x: len(acdc_dataset[x]) for x in ['train', 'val']}

    model_ft = get_model(train_args["model"], device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    model_ft = train(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                     num_epochs=100)

