import os
import argparse
import yaml
import tqdm
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


BLUE = (31, 119, 180)
YELLOW = (255, 127, 14)
GREEN = (44, 160, 44)
RED = (214, 39, 40)
PURPLE = (148, 103, 189)
BROWN = (140, 86, 75)
PINK = (227, 119, 194)
#GRAY = (128, 128, 128)
LIME = (188, 189, 34)
TEAL = (23, 25, 207)

CCYCLE = [BLUE, YELLOW, GREEN, RED, PURPLE, BROWN, PINK, LIME, TEAL]


class IndexTracker(object):
    def __init__(self, ax, X, dataset_name=None, fn=None):
        """
        ax: Axes object
        X : image in (w, h, c, t) format where c can be either 3 or 1.
        """
        self.ax = ax
        ax.set_title(dataset_name + ": " + fn)

        self.X = X
        self.rank = len(X.shape)

        if self.rank == 3:
            rows, cols, self.slices = X.shape
            self.ind = self.slices // 2
            self.im = ax.imshow(self.X[:, :, self.ind])
        elif self.rank == 4:
            rows, cols, ch, self.slices = X.shape
            self.ind = self.slices // 2
            self.im = ax.imshow(self.X[:, :, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if self.rank == 3:
            self.im.set_data(self.X[:, :, self.ind])
        elif self.rank == 4:
            self.im.set_data(self.X[:, :, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def apply_segm_mask_on_img(img, segm_mask, alpha=0.5):
    img = np.stack([img, img, img], axis=2).astype("uint8")
    segm_mask = np.stack([segm_mask, segm_mask, segm_mask], axis=2).astype("uint8")
    vals = np.unique(segm_mask)[1:]
    for d, val in enumerate(vals):
        single_color_img = create_single_color_img(CCYCLE[d], segm_mask.shape)
        segm_mask = np.where(segm_mask == val, single_color_img, segm_mask).astype("uint8")
    img = (1 - alpha) * img + segm_mask * alpha
    img = img.astype("uint8")
    return img, segm_mask


def create_single_color_img(color, size):
    img = np.ones(size)
    for c in range(img.shape[2]):
        img[:, :, c, :] *= color[c]
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nii.gz-1 visualizer')
    parser.add_argument('--yaml_path', type=str, metavar='YAML',
                        default="config/test.yaml",
                        help='Enter the path for the YAML config')
    args = parser.parse_args()

    yaml_path = args.yaml_path
    with open(yaml_path, 'r') as f:
        data_args = yaml.safe_load(f)

    example_file = os.path.join(data_args["sample_file"])

    proxy_img = nib.load(example_file)
    img = proxy_img.get_fdata()

    content = os.listdir(data_args["sample_folder"])
    files = [os.path.join(data_args["sample_folder"], x) for x in content]

    files = sorted(list(filter(lambda x: x[-7:] == ".nii.gz", files)))
    files_imgs = sorted(list(filter(lambda x: "image" in x, files)))
    files_labels = sorted(list(filter(lambda x: "label" in x, files)))

    for img_file, segm_mask_file in tqdm.tqdm(zip(files_imgs, files_labels)):
        proxy_img = nib.load(img_file)
        img = proxy_img.get_fdata()

        proxy_segm_mask = nib.load(segm_mask_file)
        segm_mask = proxy_segm_mask.get_fdata()

        img, segm_mask = apply_segm_mask_on_img(img, segm_mask)

        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, np.concatenate([img, segm_mask], axis=1))
        print(data_args["dataset_name"])
        tracker = IndexTracker(ax, img, dataset_name=data_args["dataset_name"], fn=img_file[:-7])
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

        if data_args["create_gif"]:
            def update(i):
                ax.imshow(img[:, :, :, i])
                ax.set_title("Slice {}".format(i), fontsize=12)
                ax.set_axis_off()

            t = img.shape[-1]
            anim = FuncAnimation(fig, update, frames=np.arange(0, t-1), interval=t)
            anim.save(img_file[:-7] + '.gif', dpi=80, writer='imagemagick')
        plt.show()

