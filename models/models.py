import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name, device, dtype=torch.float32, pretrained=False):
    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        #model_ft.fc = nn.Linear(num_ftrs, 10)
    elif model_name == "resnet34":
        model_ft = models.resnet34(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        #model_ft.fc = nn.Linear(num_ftrs, 10)
    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        #model_ft.fc = nn.Linear(num_ftrs, 10)
    elif model_name == "vgg11":
        model_ft = models.vgg11(pretrained=pretrained)
        model_ft.classifier[6] = nn.Linear(4096, 2, bias=True)

    model_ft = model_ft.to(dtype).to(device)
    return model_ft
