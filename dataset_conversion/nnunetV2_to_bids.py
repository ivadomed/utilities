"""
Converts nnUNetv2 dataset format to the BIDS-structured dataset. Full details about
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Théo Mathieu
"""

import argparse
import numpy as np
import shutil
import pathlib
from pathlib import Path
import datetime
import json
import os
import nibabel as nib


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--path-data', required=True, help='Path to nnUNet dataset. Example: ~/data/dataset')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-bids')
    parser.add_argument('--copy', '-cp', type=bool, default=False,
                        help='Making symlink (False) or copying (True) the files in the Bids dataset, default = False. '
                             'Example for symlink: --copy True')
    return parser


def separate_labels(label_file, original_label, dataset_label, label_new_dir, dataset_name):
    """
    Function to make one nifti file for each possible voxel value

    Args:
        label_file (str): Path to the label file ...label-
        original_label (str): Path to the label file in the nnUNetV2 dataset
        dataset_label (str): Labels keys from the dataset.json file
        label_new_dir (str): Folder for the label file in Bids format
        dataset_name (str): nnUNetV2 dataset name
    """
    # TODO for region based segmentation add the different part to make one part
    # (this issue: https://github.com/ivadomed/data-conversion/pull/15#issuecomment-1599351103)
    value_label = {v: k for k, v in dataset_label.items()}
    nifti_file = nib.load(original_label)
    data = nifti_file.get_fdata()
    for value in value_label.keys():
        if value != 0:
            seg_name = value_label[value]
            voxel_val = np.zeros_like(data)
            if type(value) == list:
                for sub_val in value:
                    voxel_val[data == sub_val] = 1
            else:
                voxel_val[data == value] = 1
            voxel_img = nib.Nifti1Image(voxel_val, nifti_file.affine, nifti_file.header)
            path_to_label = os.path.join(label_new_dir, f"{label_file}-{seg_name}_seg.nii.gz")
            nib.save(voxel_img, path_to_label)
            json_name = f"{label_file}-{seg_name}_seg.json"
            write_json(os.path.join(label_new_dir, json_name), dataset_name)


def write_json(filename, dataset_name):
    """
    Save a json file with the label image created

    Args:
        filename (str): Json filename (with path)
        dataset_name (str): Name of the dataset
    """
    data = {
        "Author": f"nnUNetV2_to_bids.py (git link?) from nnUNet dataset {dataset_name}",
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Write the data to the JSON file
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def get_subject_info(file_name, contrast_dict):
    """
    Get different information about the current subject

    Args:
        file_name (str): Filename corresponding to the subject image
        contrast_dict (dict): Dictionary, key channel_names from dataset.json

    Returns:
        sub_names (str): Name of the subject. Example: sub-milan002
        ses (str): ses name
        bids_nb (str): subject number in the BIDS dataset
        info[2] (str): Contrast value in BIDS format. Example 0001
        contrast (str): Image contrast (T2w, T1, ...)

    """
    name = file_name.split(".")[0]
    info = name.split("_")
    if len(info) == 4:
        ses = info.pop(1)
    else:
        ses = None
    sub_name = info[0]
    bids_nb = info[1]
    info[2] = info[2].lstrip('0')
    if info[2] == '':
        info[2] = '0'
    contrast = contrast_dict[info[2]]
    return sub_name, ses, bids_nb, info[2], contrast


def main():
    parser = get_parser()
    args = parser.parse_args()
    copy = args.copy
    root = Path(os.path.abspath(os.path.expanduser(args.path_data)))
    path_out = Path(os.path.abspath(os.path.expanduser(args.path_out)))
    with open(os.path.join(root, "dataset.json"), 'r') as json_file:
        dataset_info = json.load(json_file)
    for folder in [("imagesTr", "labelsTr"), ("imagesTs", "labelsTs")]:
        for image_file in os.listdir(f"{root}/{folder[0]}/"):
            sub_name, ses, bids_nb, bids_contrast, contrast = get_subject_info(image_file,
                                                                               dataset_info["channel_names"])
            # TODO separate the label file in multiple file one by integer and get the coresponding label
            #  in the dataset file
            if ses:
                image_new_dir = os.path.join(path_out, sub_name, ses, 'anat')
                label_new_dir = os.path.join(path_out, 'derivatives/labels', sub_name, ses, 'anat')
                pathlib.Path(image_new_dir).mkdir(parents=True, exist_ok=True)
                pathlib.Path(label_new_dir).mkdir(parents=True, exist_ok=True)
                bids_image_name = f"{sub_name}_{ses}_{contrast}.nii.gz"
                label_name = f"{sub_name}_{ses}_{bids_nb}.nii.gz"
                label_file = os.path.join(root, folder[1], label_name)
                separate_labels(f"{sub_name}_{ses}_{contrast}_label", label_file, dataset_info["labels"], label_new_dir,
                                str(root).split('/')[-1])
            else:
                image_new_dir = os.path.join(path_out, sub_name, 'anat')
                label_new_dir = os.path.join(path_out, 'derivatives/labels', sub_name, 'anat')
                pathlib.Path(image_new_dir).mkdir(parents=True, exist_ok=True)
                pathlib.Path(label_new_dir).mkdir(parents=True, exist_ok=True)
                bids_image_name = f"{sub_name}_{contrast}.nii.gz"
                label_name = f"{sub_name}_{bids_nb}.nii.gz"
                label_file = os.path.join(root, folder[1], label_name)
                separate_labels(f"{sub_name}_{contrast}_label", label_file, dataset_info["labels"], label_new_dir,
                                str(root).split('/')[-1])
            image_file = os.path.join(root, folder[0], image_file)
            if copy:
                shutil.copy2(os.path.abspath(image_file), os.path.join(image_new_dir, bids_image_name))
                # shutil.copy2(os.path.abspath(label_file), os.path.join(label_new_dir, bids_label_name))
            else:
                os.symlink(os.path.abspath(image_file), os.path.join(image_new_dir, bids_image_name))
                # os.symlink(os.path.abspath(label_file), os.path.join(label_new_dir, bids_label_name))


if __name__ == '__main__':
    main()
