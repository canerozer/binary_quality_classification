from pandas.core.common import flatten
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import torch


def list_subfolders(root: str) -> list:
    """
    Lists all the files given a root folder.
    Args:
        root (str): Root folder

    Returns:
        files (list): All files that are found within the folder
    """
    files = []
    content = sorted(os.listdir(root))
    for f in content:
        target = os.path.join(root, f)
        if os.path.isdir(target):
            out = list_subfolders(target)
            files.append(out)
        elif os.path.isfile(target):
                files.append(target)
        files = list(flatten(files))
    return files


def show_4d_images(fig: matplotlib.figure.Figure,
                   ax: matplotlib.axes.Axes,
                   image: np.array, est_cor_idx: list=None,
                   img_name: str=None):
    """
    Lists all the files given a root folder.
    Args:
        fig (matplotlib.figure.Figure): Figure object
        ax (matplotlib.axes.Axes): Axes object
        image (np.array): 4D image tensor to be shown
        est_cor_idx (list): List containing the percentage of corruption
                           for each time and z-axes.
    """
    assert len(image.shape) == 4, "Image's tensor rank is {}, not 4".format(
                                  len(image.shape))

    axcolor = 'lightgoldenrodyellow'
    axz = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    axt = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)

    zpos = Slider(axz, 'Z Axis', 1, image.shape[-2], valfmt="%d")
    tpos = Slider(axt, 'Time Axis', 1, image.shape[-1], valfmt="%d")

    pos_z = 0
    pos_t = 0

    im = ax.imshow(image[:, :, pos_z, pos_t])
    if est_cor_idx:
        ax.set_title("ID: " + img_name +
                     " Estimated Corruption: {:.5}".format(
                     str(est_cor_idx)))
    else:
        ax.set_title("ID: " + img_name)

    def update(val):
        pos_z = int(zpos.val)
        pos_t = int(tpos.val)

        fig.canvas.draw_idle()

        im.set_data(image[:, :, pos_z - 1, pos_t - 1])

        if est_cor_idx:
            ax.set_title("ID: " + img_name +
                         " Estimated Corruption: {:.5}".format(
                         str(est_cor_idx)))
        else:
            ax.set_title("ID: " + img_name)

    zpos.on_changed(update)
    tpos.on_changed(update)

    plt.show()


def get_most_recent_state(model_name: str, model_locs: str) -> dict:
    files = os.listdir(model_locs)
    files = list(filter(lambda x: model_name in x, files))
    files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[-1].split("ep")[-1]))
    recent_state_file = files[-1]
    state = torch.load(os.path.join(model_locs, recent_state_file))
    return state


def get_most_recent_model(model_name: str, model_locs: str) -> dict:
    files = os.listdir(model_locs)
    files = list(filter(lambda x: model_name in x, files))
    files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[-1].split("ep")[-1]))
    recent_state_file = files[-1]
    state = torch.load(os.path.join(model_locs, recent_state_file))
    return state["model"]


def yaml_var_concat(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


if __name__ == "__main__":
    state =  get_most_recent_state("resnet50", "/media/dontgetdown/model_partition/models/")
    print(state)
