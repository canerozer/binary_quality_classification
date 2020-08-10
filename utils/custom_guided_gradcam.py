"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import argparse
import yaml
import re
import os
import numpy as np
from PIL import Image

import torch

from utils.misc_functions import (get_example_params, convert_to_grayscale,
                                  preprocess_image, get_model, DictAsMember,
                                  custom_save_gradient_images, recreate_image,
                                  open_image)
from utils.custom_gradcam import GradCam
from utils.custom_guided_backprop import GuidedBackprop


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


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
        f = f.split(".npy")[0]
        """
        if "hypothetical" not in yaml_path:
            yaml_path = yaml_path.split(".npy")[0]
            c = re.findall(r'\d+.\d+', yaml_path)
            if c == []:
                c = re.findall(r'good', yaml_path)
            img.save("results/acdc/test/" + f + "_" + c[0] + ".jpg")
        """
        prep_img = preprocess_image(img, h=h, w=w)

        cam = gcv2.generate_cam(prep_img, vis_args.DATASET.target_class)
        guided_grads = GBP.generate_gradients(prep_img,
                                              vis_args.DATASET.target_class)
        cam_gb = guided_grad_cam(cam, guided_grads)

        bw_cam_gb = convert_to_grayscale(cam_gb)
        prep_img = recreate_image(prep_img)
        r = alpha * bw_cam_gb + (1 - alpha) * prep_img
        r = ((r - r.min()) / (r.max() - r.min()) * 255).astype(np.float32)
        custom_save_gradient_images(r, vis_args.RESULTS.dir, f,
                                    obj="bw_ggrad_cam")

    """
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)

    # Grad cam
    gcv2 = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = gcv2.generate_cam(prep_img, target_class)
    print('Grad cam completed')

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    print('Guided backpropagation completed')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
    """
    print('Guided grad cam completed')
