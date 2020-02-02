import os
import argparse
import yaml
import tqdm

import numpy as np
import nibabel as nib
from nibabel.spatialimages import SpatialImage

import matplotlib.pyplot as plt

from utils.transform import transform_kspace_to_image, transform_image_to_kspace
from nishow import IndexTracker


def normal_pmf(x, mean, sigma):
    """Constructs the PMF in a Gaussian shape.

    Args:
        x (np.array): Random Variables.
        mean (float): Mean of the Gaussian RV.
        sigma (float): Standard deviation of the Gaussian RV.

    Returns:
        x (np.array): PMF in a Gaussian shape given the random variables and
                      parameters.

    """
    x = np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)
    x /= np.sqrt(2 * np.pi * sigma ** 2)
    x /= x.sum()
    return x


def corrupt(kspace, sigma=1.0):
    """Corrupt a patient's MRI data using other adjacent frames. 
    This function implements the idea of the fig. 3 of 1808.05130.pdf.

    Args:
        kspace (np.array): Image in Fourier domain.
        sigma (float): Std dev of the each Gaussian component in the mixture.

    Returns:
        corrupted_kspace (np.array): Corrupted Image in Fourier domain.

    """

    corrupted_kspace = np.zeros_like(kspace)

    n_lines = kspace.shape[0]
    fr_num = kspace.shape[2]

    for t in range(fr_num, fr_num * 2):
        # Creates three gaussian pmf's and combines them with equal weights
        pmf = np.arange(fr_num * 4 + 1)
        pmf0 = normal_pmf(pmf, t, sigma=sigma)
        pmf1 = normal_pmf(pmf, t + fr_num, sigma=sigma)
        pmf2 = normal_pmf(pmf, t + fr_num * 2, sigma=sigma)
        pmf_samp = 1./3 * pmf0 + 1./3 * pmf1 + 1./3 * pmf2

        # Samples the frame numbers using the PMF of GMM
        mask = np.random.multinomial(t, pmf_samp, n_lines)
        fr_idx = np.argmax(mask, axis=1)
        fr_idx = fr_idx % fr_num

        # Transfers the line of the selected frame to the new kspace matrix
        t = t % fr_num
        for n, idx in zip(range(n_lines), fr_idx):
            corrupted_kspace[n, :, t] = kspace[n, :, idx]
    return corrupted_kspace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nii.gz-1 visualizer')
    parser.add_argument('--yaml_path', type=str, metavar='YAML',
                        default="config/test.yaml",
                        help='Enter the path for the YAML config')
    args = parser.parse_args()

    yaml_path = args.yaml_path
    with open(yaml_path, 'r') as f:
        data_args = yaml.safe_load(f)

    content = sorted(os.listdir(data_args["sample_folder"]))
    content = sorted(list(filter(lambda x: x[-7:] == ".nii.gz", content)))
    #content = sorted(list(filter(lambda x: x[-4:] == ".nii", content)))
    content = sorted(list(filter(lambda x: "image" in x, content)))
    files = [os.path.join(data_args["sample_folder"], x) for x in content]


    for sigma in data_args["sigmas"]:
        for img_name, img_file in tqdm.tqdm(zip(content, files)):
            proxy_img = nib.load(img_file)
            img = proxy_img.get_fdata()

            kspace = transform_image_to_kspace(img)

            kspace_prime = corrupt(kspace, sigma=sigma)

            recon_img_prime = transform_kspace_to_image(kspace_prime).astype("float64")

            if data_args["show"]:
                fig, ax = plt.subplots(1, 1)

                recon_img = transform_kspace_to_image(kspace).astype("float64")
                tracker = IndexTracker(ax,
                                       np.concatenate([img.astype("float64"),
                                                       recon_img_prime,
                                                       recon_img],
                                                      axis=1),
                                       dataset_name=data_args["dataset_name"],
                                       fn=img_file[:-7])
                fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
                plt.show()

            target_path = img_name[:-4]+"_sigma"+str(sigma)+img_name[-7:]
            affine = proxy_img.affine
            print(affine)
            proxy_recon_img_prime = nib.Nifti1Image(recon_img_prime, affine)
            nib.save(proxy_recon_img_prime,
                     os.path.join(data_args["save_to"], target_path),
                    )


