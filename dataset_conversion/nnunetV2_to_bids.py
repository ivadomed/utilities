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
import datetime
import json
import os


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--path-data', required=True, help='Path to nnUNet dataset. Example: ~/data/dataset')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-bids')
    parser.add_argument('--copy', '-cp', type=bool, default=False,
                        help='Making symlink (False) or copying (True) the files in the Bids dataset, default = False. '
                             'Example for symlink: --copy True')
    return parser


def separate_labels(label_file, dataset_label):
    value_label = {v: k for k, v in dataset_label.items()}
    for value in value_label.keys():
        if value != 0:
            print(value_label[value])
            #TODO create new nifti image with only this value and _label-manual.nii.gz
    return 0


def write_json(filename, dataset_name):
    data = {
        "Author": f"nnUNetV2_to_bids.py (git link?) from nnUNet dataset {dataset_name}",
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Write the data to the JSON file
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


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
    return sub_name, ses, bids_nb, info[2], contrast


def main():
    parser = get_parser()
    args = parser.parse_args()
    copy = args.copy
    root = Path(os.path.abspath(os.path.expanduser(args.path_data)))
    path_out = Path(os.path.abspath(os.path.expanduser(args.path_out)))
    with open(os.path.join(root, "dataset.json"), 'r') as json_file:
        dataset_info = json.load(json_file)
    separate_labels("_", dataset_info["labels"])
    for folder in [("imagesTr", "labelsTr"), ("imagesTs", "labelsTs")]:
        for image_file in os.listdir(f"{root}/{folder[0]}/"):
            sub_name, ses, bids_nb, bids_contrast, contrast = get_subject_info(image_file,
                                                                               dataset_info["channel_names"])
            # TODO separate the label file in multiple file one by integer and get the coresponding label
            #  in the dataset file
            if ses:
                image_new_dir = os.path.join(path_out, sub_name, ses, 'anat')
                label_new_dir = os.path.join(path_out, 'derivatives/labels', sub_name, ses, 'anat')
                bids_image_name = f"{sub_name}_{ses}_{contrast}.nii.gz"
                bids_label_name = f"{sub_name}_{ses}_{contrast}_label-manual.nii.gz"
                json_name = f"{sub_name}_{ses}_{contrast}_label-manual.json"
                label_name = f"{sub_name}_{ses}_{bids_nb}.nii.gz"
            else:
                image_new_dir = os.path.join(path_out, sub_name, 'anat')
                label_new_dir = os.path.join(path_out, 'derivatives/labels', sub_name, 'anat')
                bids_image_name = f"{sub_name}_{contrast}.nii.gz"
                bids_label_name = f"{sub_name}_{contrast}_label-manual.nii.gz"
                json_name = f"{sub_name}_{contrast}_label-manual.json"
                label_name = f"{sub_name}_{bids_nb}.nii.gz"
            image_file = os.path.join(root, folder[0], image_file)
            label_file = os.path.join(root, folder[1], label_name)
            pathlib.Path(image_new_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(label_new_dir).mkdir(parents=True, exist_ok=True)
            write_json(os.path.join(label_new_dir, json_name), str(root).split('/')[-1])
            if copy:
                shutil.copy2(os.path.abspath(image_file), os.path.join(image_new_dir, bids_image_name))
                shutil.copy2(os.path.abspath(label_file), os.path.join(label_new_dir, bids_label_name))
            else:
                os.symlink(os.path.abspath(image_file), os.path.join(image_new_dir, bids_image_name))
                os.symlink(os.path.abspath(label_file), os.path.join(label_new_dir, bids_label_name))


if __name__ == '__main__':
    main()
