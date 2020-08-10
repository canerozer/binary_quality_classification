"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import argparse
import yaml
import os
import tqdm
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.nn import Sequential
from torchvision.models.resnet import Bottleneck, BasicBlock

from utils.misc_functions import (get_example_params, DictAsMember,
                                  get_model, open_image, preprocess_image,
                                  recreate_image, show_bbox,
                                  custom_save_class_activation_images)


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, model_name, target_layer=None, target_block=None):
        self.model = model
        self.model_name = model_name
        self.target_layer = target_layer
        self.target_block = str(target_block)

        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def hook_alexnet(self, conv_output, x):
        for mpos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(mpos) == self.target_layer:
                x.register_hook(self.save_gradient)
                # Save the convolution output of the target layer
                conv_output = x  
        return conv_output, x

    def hook_resnet(self, conv_output, x):
        for mname, mmodule in self.model._modules.items():
            if mname == "avgpool":
                break
            if mname == self.target_layer and \
            isinstance(mmodule, Sequential):
                for id, block in mmodule._modules.items():
                    if id == self.target_block:
                        d = 0
                        relu = block._modules["relu"]
                        identity = x
                        for name, module in block._modules.items():
                            if name == "relu":
                                break
                            x = module(x)
                            if name == "conv3" and \
                            isinstance(block, Bottleneck):
                                x.register_hook(self.save_gradient)
                                conv_output = x
                            elif name == "conv2" and \
                            isinstance(block, BasicBlock):
                                x.register_hook(self.save_gradient)
                                conv_output = x
                            d += 1
                            if (d == len(block._modules.items()) - 1):
                                x += identity
                            if (d > 0 and d % 2 == 0):
                                x = relu(x)
                    else:
                        x = block(x)
            else:
                x = mmodule(x)
        return conv_output, x

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None

        if self.model_name == "alexnet":
            conv_output, x = self.hook_alexnet(conv_output, x)
        if "resnet" in self.model_name:
            conv_output, x = self.hook_resnet(conv_output, x)

        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # Forward pass on the classifier
        if self.model_name == "alexnet":
            x = x.view(x.size(0), -1)  # Flatten
            x = self.model.classifier(x)
        if "resnet" in self.model_name:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    #def __init__(self, model, target_layer):
    def __init__(self, model, model_args):
        self.model = model
        self.model.eval()
        self.model_name = model_args.name
        target_layer = model_args.last_layer
        target_block = model_args.last_block
        # Define extractor
        self.extractor = CamExtractor(self.model, self.model_name,
                                      target_layer=target_layer,
                                      target_block=target_block)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, C)
        device = input_image.device
        conv_output, model_output = self.extractor.forward_pass(input_image)
        pred = torch.argmax(model_output.data, dim=1)
        if target_class is None:
            target_class = pred

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.to(device)

        # Zero grads
        if self.model_name == "alexnet":
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        if "resnet" in self.model_name:
            for name, module in self.model._modules.items():
                module.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.detach().cpu().numpy()[0]

        # Get convolution outputs
        target = conv_output.data.detach().cpu().numpy()[0]

        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)

        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam, pred


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

    # Initialize GradCam and GBP
    gcv2 = GradCam(pretrained_model, vis_args.MODEL)

    # Get filenames and create absolute paths
    if os.path.isdir(vis_args.DATASET.path):
        files = os.listdir(vis_args.DATASET.path)
        paths = [os.path.join(vis_args.DATASET.path, f) for f in files]
    elif os.path.isfile(vis_args.DATASET.path):
        files = list(vis_args.DATASET.path.split("/")[-1])
        paths = [vis_args.DATASET.path]

    alpha = vis_args.RESULTS.alpha
    h = w = vis_args.DATASET.size

    preds_dict = {}

    for f, path in tqdm.tqdm(zip(files, paths)):
        img = open_image(path)
        prep_img = preprocess_image(img, h=h, w=w)
        cam, pred = gcv2.generate_cam(prep_img, vis_args.DATASET.target_class)
        if vis_args.RESULTS.DRAW_GT_BBOX.state:
            cam = show_bbox(img, cam, f, vis_args.RESULTS.DRAW_GT_BBOX.gt_src)
        img = img.resize((h, w))
        custom_save_class_activation_images(img, cam,
                                            vis_args.RESULTS.dir, f,
                                            obj="grad_cam")
        preds_dict[f] = pred

    img_names, preds = list(preds_dict.keys()), list(preds_dict.values())
    df = pd.DataFrame({"image_names": img_names, "predictions": preds})
    df.to_csv(vis_args.RESULTS.save_preds_to)
        
    """
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    """
    print('Grad cam completed')
