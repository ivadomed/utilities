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
import json
import os
import nibabel as nib

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert dataset in nnUNetV2 format to BIDS-structured dataset.')
    parser.add_argument('--path-in', required=True, help='Path to nnUNet dataset. Example: ~/data/dataset')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-bids')
    parser.add_argument('--suffix', required=True, help='Suffix of the label file Example: sub-003_T2w_SUFFIX.nii.gz')
    parser.add_argument('--copy', '-cp', type=bool, default=False,
                        help='Making symlink (False) or copying (True) the files in the Bids dataset. '
                             'This option only affects the image file, the label file is copied regardless of the '
                             ' option, default = False. Example for symlink: ø, Example for copy: --copy')
    return parser



def get_subject_info(file_name, contrast_dict):
    """
    Get different information about the current subject

    Args:
        file_name (str): Filename corresponding to the subject image
        contrast_dict (dict): Dictionary, key channel_names from dataset.json

    Returns:
        sub_names (str): Name of the subject. Example: sub-milan002
        ses (str): session name. Example: ses-01
        bids_nb (str): subject number in the BIDS dataset
        info[2] (str): Contrast value in BIDS format. Example: 0001
        contrast (str): Image contrast (T2w, T1, ...)

    """
    name = file_name.split(".")[0]
    info = name.split("_")
    bids_nb = info[-2]
    contrast = info[-1]
    if len(info) == 4:
        ses = info.pop(1)
    else:
        ses = None
    sub_name = "_".join(info[:-2])
    contrast = contrast.lstrip('0')
    if contrast == '':
        contrast = '0'
    contrast_bids = contrast_dict[contrast]
    return sub_name, ses, bids_nb, contrast, contrast_bids


def main():
    parser = get_parser()
    args = parser.parse_args()
    copy = args.copy
    suffix =args.suffix
    root = Path(os.path.abspath(os.path.expanduser(args.path_bids)))
    path_out = Path(os.path.abspath(os.path.expanduser(args.path_out)))
    with open(os.path.join(root, "dataset.json"), 'r') as json_file:
        dataset_info = json.load(json_file)
    for folder in [("imagesTr", "labelsTr"), ("imagesTs", "labelsTs")]:
        for image_file in os.listdir(f"{root}/{folder[0]}/"):
            if not image_file.startswith('.'):
                sub_name, ses, bids_nb, bids_contrast, contrast = get_subject_info(image_file,
                                                                                   dataset_info["channel_names"])
                if ses: # Multiple Session per subject
                    image_new_dir = os.path.join(path_out, sub_name, ses, 'anat')
                    label_new_dir = os.path.join(path_out, 'derivatives/labels', sub_name, ses, 'anat')
                    pathlib.Path(image_new_dir).mkdir(parents=True, exist_ok=True)
                    pathlib.Path(label_new_dir).mkdir(parents=True, exist_ok=True)
                    bids_image_name = f"{sub_name}_{ses}_{contrast}.nii.gz"
                    bids_label_name = f"{sub_name}_{ses}_{contrast}_{suffix}.nii.gz"
                    label_file = f"{sub_name}_{ses}_{bids_nb}.nii.gz"
                    old_label_dir = os.path.join(root, folder[1])
                else:
                    image_new_dir = os.path.join(path_out, sub_name, 'anat')
                    label_new_dir = os.path.join(path_out, 'derivatives/labels', sub_name, 'anat')
                    pathlib.Path(image_new_dir).mkdir(parents=True, exist_ok=True)
                    pathlib.Path(label_new_dir).mkdir(parents=True, exist_ok=True)
                    bids_image_name = f"{sub_name}_{contrast}.nii.gz"
                    bids_label_name = f"{sub_name}_{contrast}_{suffix}.nii.gz"
                    label_file = f"{sub_name}_{bids_nb}.nii.gz"
                    old_label_dir = os.path.join(root, folder[1])
                image_file = os.path.join(root, folder[0], image_file)
                label_file = os.path.join(old_label_dir, label_file)
                if copy:
                    shutil.copy2(os.path.abspath(image_file), os.path.join(image_new_dir, bids_image_name))
                    shutil.copy2(os.path.abspath(label_file), os.path.join(label_new_dir, bids_label_name))
                else:
                    os.symlink(os.path.abspath(image_file), os.path.join(image_new_dir, bids_image_name))
                    os.symlink(os.path.abspath(label_file), os.path.join(label_new_dir, bids_label_name))


if __name__ == '__main__':
    main()
