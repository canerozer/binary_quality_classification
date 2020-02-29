import os
import argparse
import yaml
import tqdm

import numpy as np


def get_patient_ids_and_classes(sp_args: dict) -> dict:
    pids_cids = {}

    src = sp_args["src"]
    metadata_file = sp_args["meta_fn"]

    pids = sorted(os.listdir(src))
    for pid in pids:
        with open(os.path.join(src, pid, metadata_file), 'r') as f:
            content = f.read().split("\n")
        for line in content:
            if "Group" in line: cid = line.split(": ")[-1]
            else: continue
        pids_cids[pid] = cid

    return pids_cids


def group_pids(pids_cids: dict) -> dict:
    cids_pids = {cid: [] for cid in pids_cids.values()}

    for pid, cid in pids_cids.items():
        cids_pids[cid].append(pid)

    return cids_pids


def splitter(cids_pids: dict, sp_args: dict) -> dict:
    splits = {"train": [], "val": [], "test": []}

    n_patients_class = len([x for x in cids_pids["NOR"]])

    n_train_class = int(sp_args["SPLIT"]["train"]["value"] * n_patients_class)
    n_val_class = int(sp_args["SPLIT"]["val"]["value"] * n_patients_class)
    n_test_class = int(sp_args["SPLIT"]["test"]["value"] * n_patients_class)

    n_samples = {"train": n_train_class * len(cids_pids.keys()),
                 "val": n_val_class * len(cids_pids.keys()),
                 "test": n_test_class * len(cids_pids.keys())}

    for cid in cids_pids.keys():
        for n in range(n_train_class):
            splits["train"].append(cids_pids[cid][n])

    for cid in cids_pids.keys():
        for n in range(n_train_class, n_train_class + n_val_class):
            splits["val"].append(cids_pids[cid][n])

    for cid in cids_pids.keys():
        for n in range(n_train_class + n_val_class, n_patients_class):
            splits["test"].append(cids_pids[cid][n])

    return splits, n_samples
    

def neutrality_splitter(splits: dict, n_samples: dict, sp_args: dict) -> dict:
    class_splits = {"train": {"plus": [], "minus": []},
                    "val": {"plus": [], "minus": []},
                    "test": {"plus": [], "minus": []}}

    for split in splits.keys():
        n_pos = n_samples[split] * sp_args["SPLIT"][split]["plus"]
        n_neg = n_samples[split] * sp_args["SPLIT"][split]["minus"]

        for d, pid in enumerate(splits[split]):
            if d < n_pos:
                class_splits[split]["plus"].append(pid)
            else: pass
        for d, pid in enumerate(reversed(splits[split])):
            if d < n_neg:
                class_splits[split]["minus"].append(pid)
            else: pass

    return class_splits


def apply_split(pos_neg_data_split: dict, sp_args: dict) -> dict:
    # Create folders for each split and corruption index for easiness
    [os.makedirs(x, exist_ok=True) for x in sp_args["TARGET"].values()]
    for cor_idx in sp_args["cor_idxes"]:
        [os.makedirs(os.path.join(x, str(cor_idx)), exist_ok=True)
         for x in sp_args["TARGET"].values()]

    # Traverse the src folders
    for cor_idx in tqdm.tqdm(sp_args["cor_idxes"]):
        src_pos = sp_args["pos_path"]
        src_neg = os.path.join(sp_args["neg_path"], str(cor_idx))

        content_pos = os.listdir(src_pos)    
        content_neg = os.listdir(src_neg)    

        for split in pos_neg_data_split.keys():
            for sign in pos_neg_data_split[split].keys():
                dst_sign = os.path.join(sp_args["TARGET"][split + "_" + sign],
                                        str(cor_idx))
                if sign == "plus":
                    for file in content_pos:
                        for pid in pos_neg_data_split[split][sign]:
                            if pid in file:
                                src = src_pos + "/" + file
                                dst = dst_sign + "/" + file
                                os.symlink(src, dst)
                if sign == "minus":
                    for file in content_neg:
                        for pid in pos_neg_data_split[split][sign]:
                            if pid in file:
                                src = src_neg + "/" + file
                                dst = dst_sign + "/" + file
                                os.symlink(src, dst)


    #os.symlink(src, dst)



def main():
    np.random.seed(1773)

    parser = argparse.ArgumentParser(description='Split data for quality'
                                                 'classification')
    parser.add_argument('--yaml_path', type=str, metavar='YAML',
                        default="config/split_data.yaml",
                        help='Enter the path for the YAML config')
    args = parser.parse_args()

    yaml_path = args.yaml_path
    with open(yaml_path, 'r') as f:
        sp_args = yaml.safe_load(f)

    patients_classes = get_patient_ids_and_classes(sp_args)
    groupped_patients = group_pids(patients_classes)

    data_split, n_samples = splitter(groupped_patients, sp_args)
    pos_neg_data_split = neutrality_splitter(data_split, n_samples, sp_args)

    apply_split(pos_neg_data_split, sp_args)
       

if __name__ == "__main__":
    main()
