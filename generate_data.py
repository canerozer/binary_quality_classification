import os
import argparse
import yaml
import tqdm
import skimage.io as io

import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from utils.utils import list_subfolders, show_4d_images, Sampler
from utils.transform import transform_kspace_to_image, transform_image_to_kspace


def normal_pmf(x: np.array,
               mean: float, sigma: float,
               reduce: bool) -> np.array:
    """Constructs the PMF in a Gaussian shape.

    Args:
        x (np.array): Random Variables.
        mean (float): Mean of the Gaussian RV.
        sigma (float): Standard deviation of the Gaussian RV.
        reduce (bool): Assign the PMF of the mean value to 0.

    Returns:
        x (np.array): PMF in a Gaussian shape given the random variables and
                      parameters.

    """
    x = np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)
    x /= np.sqrt(2 * np.pi * sigma ** 2)
    if reduce:
       x[mean] = 0.
    x /= x.sum()
    return x


def corrupt_slices(kspace: np.array, data_args: dict, 
                   sigma: float=1.0,
                   cor_idx: float= 0.) -> np.array:
    """Corrupt a patient's MRI data using other adjacent slices. 
    This function implements the idea of the fig. 3 of 1808.05130.pdf.

    Args:
        kspace (np.array): Image in Fourier domain.
        data_args (dict): Properties regarding data generation from YAML file
        sigma (float): Std dev of the each Gaussian component in the mixture.
        cor_idx (float): Corruption Index.

    Returns:
        corrupted_kspace (np.array): Corrupted Image in Fourier domain.
        est_cor_idx (float): Estimated Corruption Index
    """

    corrupted_kspace = np.copy(kspace)
    est_cor_idx = 0.

    n_lines = kspace.shape[0]
    z_axis = kspace.shape[2]
    fr_num = kspace.shape[3]
    n_chg_lines = int(n_lines * cor_idx)
    change_every = int(1 / cor_idx)
    est_cor_idx = n_chg_lines / n_lines

    for z in range(z_axis):
        for t in range(fr_num):
            if data_args["sampling_type"] == "uniform":
                line_ids_changed = np.random.choice(n_lines,
                                                    size=n_chg_lines,
                                                    replace=False)
            elif data_args["sampling_type"] == "regular":
                line_ids_changed = list(range(0, n_lines - 1, change_every))
            elif data_args["sampling_type"] == "inv_gaussian":
                line_ids_changed = np.random.choice(n_lines,
                                                    size=n_chg_lines,
                                                    replace=False,
                                                    p=p)
                raise NotImplementedError

            # Create a generic Gaussian PMF
            rvs = np.arange(fr_num)
            pmf = normal_pmf(rvs, len(rvs) // 2, sigma=sigma, reduce=True)

            # Shift the positions of the random variable dependent to the mean
            if t <= fr_num // 2:
                shifted_rvs = list(range(len(rvs) // 2 + t, len(rvs))) +\
                              list(range(0, len(rvs) // 2 + t))
            elif t >= fr_num // 2:
                shifted_rvs = list(range(t - len(rvs) // 2, len(rvs))) +\
                              list(range(0, t - len(rvs) // 2))

            # Sample the indexes based on the shifted RVs
            final_idxes = np.random.choice(shifted_rvs,
                                           size=len(line_ids_changed),
                                           replace=True,
                                           p=pmf)

            # Corrupt the lines of k-space 
            for n, idx in zip(line_ids_changed, final_idxes):
                corrupted_kspace[n, :, z, t] = kspace[n, :, z, idx]

    return corrupted_kspace, est_cor_idx


if __name__ == "__main__":
    plt.set_cmap("gray")
    np.random.seed(1773)

    parser = argparse.ArgumentParser(description='NifTI-2 corrupted data gen')
    parser.add_argument('--yaml_path', type=str, metavar='YAML',
                        default="config/construct_data.yaml",
                        help='Enter the path for the YAML config')
    args = parser.parse_args()

    yaml_path = args.yaml_path
    with open(yaml_path, 'r') as f:
        data_args = yaml.safe_load(f)

    if os.path.isdir(data_args["src"]):
        content = list_subfolders(data_args["src"])

        length = len(data_args["format"])
        content = list(filter(lambda x: x[-length:] == data_args["format"],
                              content))

        content = list(filter(lambda x: data_args["process_ones_with"] in x,
                              content))

        files = [os.path.join(data_args["src"], x) for x in content]
        content = [x.split("/")[-1].split("_")[0] for x in content]
    elif os.path.isfile(data_args["src"]):
        content = [data_args["src"].split("/")[-1]]
        files = [data_args["src"]]
    else:
        raise FileNotFoundError("No such file or directory: {}".format(\
              data_args["src"]))

    # Save Positive Samples
    if data_args["save"]:
        for img_name, img_file in tqdm.tqdm(zip(content, files)):
            proxy_img = nib.load(img_file)
            img = proxy_img.get_fdata().astype("float32")

            os.makedirs(data_args["save_pos"], exist_ok=True)
            for z_ax in range(img.shape[2]):
                for frame_nr in range(img.shape[3]):
                    target_path_pos = img_name + "_z" +\
                                      str(z_ax) + "_fn" +\
                                      str(frame_nr)# + ".png"
                    np.save(os.path.join(data_args["save_pos"], target_path_pos),
                              img[:, :, z_ax, frame_nr])

    # Create and Save Negative Samples
    for cor_idx in data_args["cor_idxes"]:
        for sigma in data_args["sigmas"]:
            for img_name, img_file in tqdm.tqdm(zip(content, files)):
                proxy_img = nib.load(img_file)
                img = proxy_img.get_fdata()
                kspace = transform_image_to_kspace(img)

                kspace_prime, est_cor_idx = corrupt_slices(kspace,
                                                          data_args,
                                                          sigma=sigma,
                                                          cor_idx=cor_idx)

                recon_img = transform_kspace_to_image(kspace)
                recon_img_prime = transform_kspace_to_image(kspace_prime)

                recon_img = recon_img.astype("float32")
                recon_img_prime = recon_img_prime.astype("float32")

                if data_args["show_image"]:
                    fig, ax = plt.subplots(1, 1)

                    X = np.concatenate([img.astype("float32"),
                                        recon_img_prime,
                                        recon_img], axis=1)
                    show_4d_images(fig, ax, X,
                                   est_cor_idx=est_cor_idx, img_name=img_name)

                if data_args["show_kspace"]:
                    fig, ax = plt.subplots(1, 1)

                    X = np.concatenate([np.abs(kspace),
                        np.abs(kspace_prime),
                        np.abs(kspace - kspace_prime)], axis=1)
                    show_4d_images(fig, ax, X,
                                   est_cor_idx=est_cor_idx, img_name=img_name)

                if data_args["save"]:
                    os.makedirs(data_args["save_neg"], exist_ok=True)
                    os.makedirs(data_args["save_neg"] +
                                str(cor_idx), exist_ok=True)
                    for z_ax in range(img.shape[2]):
                        for frame_nr in range(img.shape[3]):
                            target_path_neg = img_name + "_sigma" +\
                                              str(sigma) + "_z" +\
                                              str(z_ax) + "_fn" +\
                                              str(frame_nr)# + ".png"
                            np.save(os.path.join(data_args["save_neg"],
                                                 str(cor_idx),
                                                 target_path_neg),
                                    recon_img_prime[:, :, z_ax, frame_nr])
                            #io.imsave(os.path.join(data_args["save_neg"],
                            #                       target_path_neg),
                            #          recon_img_prime[:, :, z_ax, frame_nr])
                            
                #for frame_nr in range(img.shape[2]):
                #    target_path_neg = img_name[:-4] + "_sigma" + str(sigma) +\
                #                      "_" + str(frame_nr)
                #    target_path_pos = img_name[:-4] + "_" + str(frame_nr)
                #    np.save(os.path.join(data_args["save_neg"], target_path_neg),
                #            recon_img_prime[:, :, frame_nr])
                #    np.save(os.path.join(data_args["save_pos"], target_path_pos),
                #            img[:, :, frame_nr])


