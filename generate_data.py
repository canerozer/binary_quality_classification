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
from nishow import IndexTracker


def normal_pmf(x: np.array, mean: float, sigma: float) -> np.array:
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


def reduced_normal_pmf(x: np.array, mean: float, sigma: float) -> np.array:
    """Constructs the PMF in a Gaussian shape.
    PMF value of the mean value has been assigned to 0.

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
    x[mean] = 0.
    x /= x.sum()
    return x


def corrupt_slices(kspace: np.array, data_args: dict, 
                   sigma: float=1.0,
                   cor_pc: float= 0.) -> np.array:
    """Corrupt a patient's MRI data using other adjacent slices. 
    This function implements the idea of the fig. 3 of 1808.05130.pdf.

    Args:
        kspace (np.array): Image in Fourier domain.
        data_args (dict): Properties regarding data generation from YAML file
        sigma (float): Std dev of the each Gaussian component in the mixture.
        cor_pc (float): Percentage of the lines that will be changed.

    Returns:
        corrupted_kspace (np.array): Corrupted Image in Fourier domain.

    """

    corrupted_kspace = np.zeros_like(kspace)
    est_cor_pcs = []

    n_lines = kspace.shape[0]
    z_axis = kspace.shape[2]
    fr_num = kspace.shape[3]
    n_chg_lines = int(n_lines * cor_pc)
    z = int(1 / cor_pc)

    for z in range(z_axis):
        for t in range(fr_num, fr_num * 2):
            if data_args["sampling_type"] == "uniform":
                line_ids_changed = np.random.choice(n_lines,
                                                    size=n_chg_lines,
                                                    replace=False)
            elif data_args["sampling_type"] == "cartesian":
                line_ids_changed = list(range(0, n_lines - 1, z))
            elif data_args["sampling_type"] == "inv_gaussian":
                line_ids_changed = np.random.choice(n_lines,
                                                    size=n_chg_lines,
                                                    replace=False,
                                                    p=p)

            rvs = np.arange(fr_num)
            # # Create 3 gaussian pmf's and combine them with equal weights
            # # PMF's are designed to sample values other than the mean value.
            # pmf = np.arange(fr_num * 4 + 1)
            # pmf0 = reduced_normal_pmf(pmf, t, sigma=sigma)
            # pmf1 = reduced_normal_pmf(pmf, t + fr_num, sigma=sigma)
            # pmf2 = reduced_normal_pmf(pmf, t + fr_num * 2, sigma=sigma)
            # pmf_samp = 1./3 * pmf0 + 1./3 * pmf1 + 1./3 * pmf2
            # if z == 6 and t == 11 + 30:
            #     fig, ax = plt.subplots()
            #     ax.plot(pmf_samp)
            #     ax.vlines(x=fr_num, ymin=-0.01, ymax=0.15, linestyles="dashed")
            #     ax.vlines(x=fr_num * 2, ymin = -0.01, ymax=0.15, linestyles="dashed")
            #     plt.show()

            # # Create a binary mask to select the lines that will be changed
            # mask = np.random.binomial(1, cor_pc, n_lines)
            # est_cor_pc = sum(mask) / len(mask)
            # est_cor_pcs_z.append(est_cor_pc)

            # # Retrieve the List of line numbers
            # lines = np.array([t % fr_num] * n_lines)

            # # Samples the frame numbers using the PMF of GMM
            # rvs = np.random.multinomial(t, pmf_samp, n_lines)
            # fr_idx = np.argmax(rvs, axis=1)
            # fr_idx = fr_idx % fr_num

            # # Gather the elements from fr_idx and fr_nums based on the mask
            # final_idxes = np.where(mask == 1, fr_idx, lines)

            # # Transfers the line of the selected frame to new k-space
            # t = t % fr_num
            # for n, idx in zip(range(n_lines), final_idxes):
            #     corrupted_kspace[n, :, z, t] = kspace[n, :, z, idx]

    return corrupted_kspace, est_cor_pcs


if __name__ == "__main__":
    plt.set_cmap("gray")
    np.random.seed(0)

    parser = argparse.ArgumentParser(description='NifTI-2 visualizer')
    parser.add_argument('--yaml_path', type=str, metavar='YAML',
                        default="config/construct_data.yaml",
                        help='Enter the path for the YAML config')
    args = parser.parse_args()

    yaml_path = args.yaml_path
    with open(yaml_path, 'r') as f:
        data_args = yaml.safe_load(f)

    if os.path.isdir(data_args["src"]):
        #content = sorted(os.listdir(data_args["src"]))
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
    for cor_pc in data_args["cor_pcs"]:
        for sigma in data_args["sigmas"]:
            for img_name, img_file in tqdm.tqdm(zip(content, files)):
                proxy_img = nib.load(img_file)
                img = proxy_img.get_fdata()
                kspace = transform_image_to_kspace(img)

                kspace_prime, est_cor_pc = corrupt_slices(kspace,
                                                          data_args,
                                                          sigma=sigma,
                                                          cor_pc=cor_pc)

                recon_img = transform_kspace_to_image(kspace)
                recon_img_prime = transform_kspace_to_image(kspace_prime)

                recon_img = recon_img.astype("float32")
                recon_img_prime = recon_img_prime.astype("float32")

                if data_args["show_image"]:
                    fig, ax = plt.subplots(1, 1)

                    X = np.concatenate([img.astype("float32"),
                                        # recon_img_prime,
                                        recon_img], axis=1)
                    show_4d_images(fig, ax, X,
                                   est_cor_pc=est_cor_pc, img_name=img_name)

                if data_args["show_kspace"]:
                    fig, ax = plt.subplots(1, 1)

                    X = np.concatenate([np.abs(kspace),
                        np.abs(kspace_prime),
                        np.abs(kspace - kspace_prime)], axis=1)
                    show_4d_images(fig, ax, X,
                                   est_cor_pc=None, img_name=img_name)

                if data_args["save"]:
                    os.makedirs(data_args["save_neg"], exist_ok=True)
                    for z_ax in range(img.shape[2]):
                        for frame_nr in range(img.shape[3]):
                            target_path_neg = img_name + "_corpc" +\
                                              str(cor_pc) + "_sigma" +\
                                              str(sigma) + "_z" +\
                                              str(z_ax) + "_fn" +\
                                              str(frame_nr)# + ".png"
                            np.save(os.path.join(data_args["save_neg"],
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


