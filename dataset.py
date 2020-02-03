import os

import numpy as np
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils



class ACDCDataset(Dataset):
    def __init__(self, plus_path, minus_path, transform=None):
        self.ppath = plus_path
        self.mpath = minus_path
        self.transform = transform
        self.construct_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img = np.load(self.samples[idx])
        img = (img - np.min(img)) / 65536
        label = np.array(self.label[idx])

        if self.transform:
            img, label = self.transform([img, label])
        return img, label

    def construct_dataset(self):
        pfiles = sorted(os.listdir(self.ppath))
        mfiles = sorted(os.listdir(self.mpath))
        self.psamples = [os.path.join(self.ppath, x) for x in pfiles]
        self.msamples = [os.path.join(self.mpath, x) for x in mfiles]
        self.samples = [*self.psamples, *self.msamples]
        self.label = [1] * len(self.psamples) + [0] * len(self.msamples)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, input):
        image, label = input

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img, label


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, input):
        image, label = input

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, input):
        image, label = input

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image), torch.from_numpy(label)

