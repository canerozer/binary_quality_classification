"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import argparse
import yaml
import os
import numpy as np
from PIL import Image

import torch
from torch.nn import ReLU, Sequential
from torchvision.models.resnet import BasicBlock, Bottleneck
from utils.misc_functions import (get_example_params,
                                  convert_to_grayscale,
                                  save_gradient_images,
                                  get_positive_negative_saliency,
                                  DictAsMember, get_model, open_image,
                                  preprocess_image,
                                  custom_save_gradient_images)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, name):
        self.model = model
        self.name = name

        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Gather the first layer depending on the architecture
        if "alexnet" in self.name:
            first_layer = list(self.model.features._modules.items())[0][1]
        elif "resnet" in self.name:
            first_layer = self.model.conv1
        else:
            raise NotImplementedError

        # Register hook to the first layer
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
        Updates relu activation functions so that
            1- stores output in forward pass
            2- imputes zero for gradient values that are less than zero
        """
        # Loop through layers, hook up ReLUs
        if "alexnet" in self.name:
            self.hook_alexnet_relu()
        elif "resnet" in self.name:
            self.hook_resnet_relu()
        else:
            raise NotImplementedError

    def relu_bw_hook_f(self, module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero
        """
        # Get last forward output
        corresponding_forward_output = self.forward_relu_outputs[-1]
        corresponding_forward_output[corresponding_forward_output > 0] = 1
        modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
        del self.forward_relu_outputs[-1]  # Remove last forward output
        return (modified_grad_out, )

    def relu_fw_hook_f(self, module, ten_in, ten_out):
        """
        Store results of forward pass
        """
        self.forward_relu_outputs.append(ten_out)

    def hook_alexnet_relu(self):
        relu_bw_hook_f = self.relu_bw_hook_f
        relu_fw_hook_f = self.relu_fw_hook_f

        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_bw_hook_f)
                module.register_forward_hook(relu_fw_hook_f)

    def hook_resnet_relu(self):
        relu_bw_hook_f = self.relu_bw_hook_f
        relu_fw_hook_f = self.relu_fw_hook_f

        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_bw_hook_f)
                module.register_forward_hook(relu_fw_hook_f)
            elif isinstance(module, Sequential):
                for bpos, block in module._modules.items():
                    if isinstance(block, (Bottleneck, BasicBlock)):
                        for lpos, layer in block._modules.items():
                            if isinstance(layer, ReLU):
                                layer.register_backward_hook(relu_bw_hook_f)
                                layer.register_forward_hook(relu_fw_hook_f)

    def generate_gradients(self, input_image, target_class):
        device = input_image.device
        # Forward pass
        model_output = self.model(input_image)
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.to(device)
        # Zero gradients
        self.model.zero_grad()
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        self.forward_relu_outputs = []
        return gradients_as_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Guided GradCAM')
    parser.add_argument('--yaml', type=str, metavar="YAML",
                        default="configs/R50_LVOT.yaml",
                        help='Enter the path for the YAML config')
    parser.add_argument('--img', type=str, metavar='I', default=None,
                        help="Enter the image path")
    parser.add_argument('--target', type=int, metavar='T', default=None,
                        help="Enter the target class ID")
    args = parser.parse_args()

    yaml_path = args.yaml
    with open(yaml_path, 'r') as f:
        vis_args = DictAsMember(yaml.safe_load(f))

    if args.img:
        vis_args.DATASET.path = args.img
        vis_args.DATASET.target_class = args.target

    # Load model & pretrained params
    pretrained_model = get_model(vis_args.MODEL)
    state = torch.load(vis_args.MODEL.path)
    try:
        pretrained_model.load_state_dict(state["model"])
    except KeyError as e:
        pretrained_model.load_state_dict(state)

    # Initialize GBP
    GBP = GuidedBackprop(pretrained_model, vis_args.MODEL.name)

    # Get filenames and create absolute paths
    if os.path.isdir(vis_args.DATASET.path):
        files = os.listdir(vis_args.DATASET.path)
        paths = [os.path.join(vis_args.DATASET.path, f) for f in files]
    elif os.path.isfile(vis_args.DATASET.path):
        files = list(vis_args.DATASET.path.split("/")[-1])
        paths = [vis_args.DATASET.path]

    alpha = vis_args.RESULTS.alpha
    h = w = vis_args.DATASET.size
    for f, path in zip(files, paths):
        img = open_image(path)
        prep_img = preprocess_image(img, h=h, w=w)
        guided_grads = GBP.generate_gradients(prep_img,
                                              vis_args.DATASET.target_class)

        pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        bw_pos_sal = convert_to_grayscale(pos_sal)
        bw_neg_sal = convert_to_grayscale(neg_sal)
        prep_img = prep_img[0].mean(dim=0, keepdim=True)
        r_pos = alpha * bw_pos_sal + (1 - alpha) * prep_img.detach().numpy()
        r_neg = alpha * bw_neg_sal + (1 - alpha) * prep_img.detach().numpy()
        custom_save_gradient_images(r_pos, vis_args.RESULTS.dir, f,
                                    obj="bw_guided_backprop_pos")
        custom_save_gradient_images(r_neg, vis_args.RESULTS.dir, f,
                                    obj="bw_guided_backprop_neg")
    """
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')
    """
