"""
Converts nnUNetv2 dataset format to the BIDS-structured dataset. Full details about
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Théo Mathieu
"""

import re
import argparse
import shutil
import pathlib
from pathlib import Path
import json
import os
from collections import OrderedDict
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import nibabel as nib
import numpy as np


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--path-data', required=True, help='Path to nnUNet dataset. Example: ~/data/dataset')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-bids')
    parser.add_argument('--copy', '-cp', type=bool, default=False,
                        help='Making symlink (False) or copying (True) the files in the Bids dataset, default = False. '
                             'Example for symlink: --copy True')
    return parser


def convert_subject():
    return 0


def get_subject_info(file_name, contrast_dict):
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
    return sub_name,ses, bids_nb, info[2], contrast


def main():
    parser = get_parser()
    args = parser.parse_args()
    copy = args.copy
    root = Path(os.path.abspath(os.path.expanduser(args.path_data)))
    path_out = Path(os.path.abspath(os.path.expanduser(args.path_out)))
    with open(os.path.join(root, "dataset.json"), 'r') as json_file:
        dataset_info = json.load(json_file)
    print(dataset_info)
    for f in os.listdir(f"{root}/imagesTr/"):
        sub_name, ses, bids_nb, bids_contrast, contrast = get_subject_info(f, dataset_info["channel_names"])
        if ses:
            image_new_dir = os.path.join(path_out, sub_name, ses, 'anat')
            label_new_dir = os.path.join(path_out, 'derivatives/labels', sub_name, ses, 'anat')
            bids_image_name = f"{sub_name}_{ses}_{contrast}.nii.gz"
            bids_label_name = f"{sub_name}_{contrast}_label-manual.nii.gz"
            label_name = f"{sub_name}_{ses}_{bids_nb}.nii.gz"
        else:
            image_new_dir = os.path.join(path_out, sub_name, 'anat')
            label_new_dir = os.path.join(path_out, 'derivatives/labels', sub_name, 'anat')
            bids_image_name = f"{sub_name}_{contrast}.nii.gz"
            bids_label_name = f"{sub_name}_{contrast}_label-manual.nii.gz"
            label_name = f"{sub_name}_{bids_nb}.nii.gz"
        pathlib.Path(image_new_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(label_new_dir).mkdir(parents=True, exist_ok=True)
        if copy:
            print("copy")
        else:
            os.symlink(os.path.abspath(f),os.path.join(image_new_dir,bids_image_name))
            os.symlink(os.path.abspath(f), os.path.join(label_new_dir, bids_label_name))


if __name__ == '__main__':
    main()
