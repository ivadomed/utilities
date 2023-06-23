"""
Converts BIDS-structured dataset to the nnUNetv2 dataset format. Full details about
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Usage example:
    python convert_bids_to_nnUNetv2.py --path-data ~/data/dataset --path-out ~/data/dataset-nnunet
                    --dataset-name MyDataset --dataset-number 501 --split 0.6 0.2 --seed 99 --copy False

Naga Karthik, Jan Valosek modified by Théo Mathieu
"""
import re
import argparse
import datetime
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
    parser.add_argument('--path-data', required=True, help='Path to BIDS dataset. Example: ~/data/dataset')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-nnunet')
    parser.add_argument('--contrast', required=True, type=str, nargs="+",
                        help='Subject contrast unique or multi contrast (separated with space). Example: T2w or '
                             'acq-sag_T2w')
    parser.add_argument('--label-suffix', type=str,
                        help='Label suffix. Example: lesion-manual or seg-manual, if None no label used')
    parser.add_argument('--softseg', nargs='+', type=float, help='Voxel value class name (separated with space).'
                                                                 'If the label file is a soft segmentation, '
                                                                 'voxel value will be disctretize in class. Example: --softseg 0.001 0.25 0.5 0.75  '
                                                                 '(4 class with values [0.001, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1) respectivly')
    parser.add_argument('--dataset-name', '-dname', default='MyDataset', type=str,
                        help='Specify the task name. Example: MyDataset')
    parser.add_argument('--dataset-number', '-dnum', default=501, type=int,
                        help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')
    parser.add_argument('--seed', default=99, type=int,
                        help='Seed to be used for the random number generator split into training and test sets.')
    # argument that accepts a list of floats as train val test splits
    parser.add_argument('--split', nargs='+', type=float, default=[0.8, 0.2],
                        help='Ratios of training (includes validation) and test splits lying between 0-1. '
                             'Example: --split 0.8 0.2')
    parser.add_argument('--copy', '-cp', type=bool, default=False,
                        help='Making symlink (False) or copying (True) the files in the nnUNet dataset, '
                             'default = False. Example for symlink: --copy True')
    return parser


# TODO: Add function to convert float softseg by discretizing voxel value to integer class
def convert_subject(root, subject, channel, contrast, label_suffix, path_out_images, path_out_labels, counter,
                    list_images, list_labels, is_ses, copy, DS_name, is_softseg, session=None):
    """Function to get image from original BIDS dataset modify if needed and place
        it with a compatible name in nnUNet dataset.

    Args:
        root (str): Path to BIDS dataset directory.
        subject (str): Subject name.
        contrast (str): Type of contrast.
        label_suffix (str): suffix of the label in derivatives.
        path_out_images (str): path to the images directory in the new dataset (test or train).
        path_out_labels (str): path to the labels directory in the new dataset (test or train).
        counter (int): counter for iterating through the number of subjects.
        list_images (list): List containing the paths of training/testing images in the nnUNetv2 format.
        list_labels (list): List containing the paths of training/testing labels in the nnUNetv2 format.
        is_ses (bool): Whether or not the dataset has ses folders for each subject.
        session (str): Session name or None if dataset without session.
        copy (bool): The files in the nnUNet dataset need to be symlink of copy file (False: symlink, True: copy).
        DS_name (str): Dataset name.
        channel (int): Contrast value as integer compatible with nnUNet documentation (ex: T1 = 1, T2 = 2, FLAIR = 3).
        is_softseg (None or list): The label file have softseg values (if None label is not softseg)

    Returns:
        list_images (list): List containing the paths of training/testing images in the nnUNetv2 format.
        list_labels (list): List containing the paths of training/testing labels in the nnUNetv2 format.

    """
    if is_softseg:
        label_start_path = os.path.join(root, 'derivatives', 'labels_softseg', subject)
    else:
        label_start_path = os.path.join(root, 'derivatives', 'labels', subject)
    if is_ses:
        subject_image_file = os.path.join(root, subject, session, 'anat', f"{subject}_{session}_{contrast}.nii.gz")

        subject_label_file = os.path.join(label_start_path, session, 'anat',
                                          f"{subject}_{session}_{contrast}_{label_suffix}.nii.gz")
        sub_name = re.match(r'^([^_]+_[^_]+)', Path(subject_image_file).name).group(1)
    else:
        subject_image_file = os.path.join(root, subject, 'anat', f"{subject}_{contrast}.nii.gz")
        subject_label_file = os.path.join(label_start_path, 'anat',
                                          f"{subject}_{contrast}_{label_suffix}.nii.gz")
        sub_name = re.match(r'^([^_]+)', Path(subject_image_file).name).group(1)

    if os.path.exists(subject_image_file):
        if label_suffix is not None:
            if os.path.exists(subject_label_file):
                subject_label_file_nnunet = os.path.join(path_out_labels, f"{DS_name}-{sub_name}_{counter:03d}.nii.gz")
                list_labels.append(subject_label_file_nnunet)
                # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                subject_image_file_nnunet = os.path.join(path_out_images,
                                                         f"{DS_name}-{sub_name}_{counter:03d}_{channel:04d}.nii.gz")
                list_images.append(subject_image_file_nnunet)
                # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                if copy:
                    shutil.copy2(os.path.abspath(subject_label_file), subject_label_file_nnunet)
                    shutil.copy2(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                else:
                    os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)
                    os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)

            else:
                print(f"Label {subject_label_file} for image {subject_image_file} does not exist this {sub_name} is ignored")
    else:
        print(f"contrast {contrast} for subject {sub_name} does not exist this contrast is ignored")

    return list_images, list_labels


def discretise_soft_seg(label_file, interval):
    nifti_file = nib.load(label_file)
    data = nifti_file.get_fdata()
    class_voxel = np.zeros_like(data, dtype=np.int16)
    # TODO max intervals
    for i, value in enumerate(interval):
        lower_bound = interval[i]
        if i == len(interval) - 1:
            class_voxel[data >= lower_bound] = i + 1
        else:
            upper_bound = interval[i + 1]
            class_voxel[(data >= lower_bound) & (data < upper_bound)] = i + 1
    voxel_img = nib.Nifti1Image(class_voxel, nifti_file.affine, nifti_file.header)
    nib.save(voxel_img, label_file)
    return 0


def main():
    parser = get_parser()
    args = parser.parse_args()
    copy = args.copy
    softseg = args.softseg
    if softseg:
        copy = True
    DS_name = args.dataset_name
    contrast = args.contrast
    root = Path(os.path.abspath(os.path.expanduser(args.path_data)))
    path_out = Path(os.path.join(os.path.abspath(os.path.expanduser(args.path_out)),
                                 f'Dataset{args.dataset_number:03d}_{args.dataset_name}'))
    # Get filename
    contrast_list = args.contrast
    channel_dict = {}
    for i, contrast in enumerate(contrast_list):
        channel_dict[contrast] = i

    label_suffix = args.label_suffix
    if label_suffix is None:
        print(f"No suffix label provided, ignoring label to create this dataset")

    # create individual directories for train and test images and labels
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

    train_images, train_labels, train_masks, test_images, test_labels, test_masks = [], [], [], [], [], []

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    # set the random number generator seed
    rng = np.random.default_rng(args.seed)

    # Get all subjects from participants.tsv
    subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
    subjects = subjects_df['participant_id'].values.tolist()
    logger.info(f"Total number of subjects in the dataset: {len(subjects)}")

    # Get the training and test splits

    train_ratio, test_ratio = args.split[0], args.split[1]
    if test_ratio == 1:
        test_subjects = subjects
        train_subjects = []
    elif train_ratio == 1:
        train_subjects = subjects
        test_subjects = []
    else:
        train_subjects, test_subjects = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
        rng.shuffle(train_subjects)

    # Initialize counters for train and test subjects
    train_ctr, test_ctr = 0, 0
    # Loop through all subjects
    # TODO try to avoid duplication
    for subject in subjects:

        # Train subjects
        if subject in train_subjects:

            # Session folder(s) exist
            # Check if session folder(s) exist
            if any('ses' in folder for folder in os.listdir(os.path.join(root, subject))):
                # Get all sessions for the subject
                sessions = os.listdir(os.path.join(root, subject))
                sessions.sort()

                for session in sessions:
                    train_ctr = len(train_images)
                    for contrast in contrast_list:
                        train_images, train_labels = convert_subject(root, subject, channel_dict[contrast], contrast,
                                                                     label_suffix, path_out_imagesTr, path_out_labelsTr,
                                                                     train_ctr + test_ctr, train_images, train_labels,
                                                                     True, copy, DS_name, softseg, session)


            # No session folder(s) exist
            else:
                train_ctr = len(train_images)
                for contrast in contrast_list:
                    train_images, train_labels = convert_subject(root, subject, channel_dict[contrast], contrast,
                                                                 label_suffix, path_out_imagesTr, path_out_labelsTr,
                                                                 train_ctr + test_ctr, train_images, train_labels,
                                                                 False, copy, DS_name, softseg)

        # Test subjects
        elif subject in test_subjects:
            # Session folder(s) exist
            # Check if session folder(s) exist
            if any('ses' in folder for folder in os.listdir(os.path.join(root, subject))):
                # Get all sessions for the subject
                sessions = os.listdir(os.path.join(root, subject))
                sessions.sort()

                for session in sessions:
                    test_ctr = len(test_images)
                    for contrast in contrast_list:
                        test_images, test_labels = convert_subject(root, subject, channel_dict[contrast], contrast,
                                                                   label_suffix, path_out_imagesTs, path_out_labelsTs,
                                                                   train_ctr + test_ctr, test_images, test_labels, True,
                                                                   copy, DS_name, softseg, session)


            # No session folder(s) exist
            else:
                test_ctr = len(test_images)
                for contrast in contrast_list:
                    test_images, test_labels = convert_subject(root, subject, channel_dict[contrast], contrast,
                                                               label_suffix, path_out_imagesTs, path_out_labelsTs,
                                                               train_ctr + test_ctr, test_images, test_labels, False,
                                                               copy, DS_name, softseg)


        else:
            print("Skipping file, could not be located in the Train or Test splits split.", subject)
    if softseg:
        for file in train_labels + test_labels:
            discretise_soft_seg(file, softseg)

    logger.info(f"Number of training and validation subjects (including sessions): {train_ctr}")
    logger.info(f"Number of test subjects (including sessions): {test_ctr}")
    # assert train_ctr == len(train_subjects), 'No. of train/val images do not match'
    # assert test_ctr == len(test_subjects), 'No. of test images do not match'

    # c.f. dataset json generation
    # In nnUNet V2, dataset.json file has become much shorter. The description of the fields and changes
    # can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson
    # this file can be automatically generated using the following code here:
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/generate_dataset_json.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()

    # The following keys are the most important ones.
    """
    channel_names:
        Channel names must map the index to the name of the channel. For BIDS, this refers to the contrast suffix.
        {
            "0": "FLAIR",
            "1": "T1w",
            "2": "T2",
            "3": "T2w"
        }
    Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based 
        training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training: 
        https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }
        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!
    """

    json_dict['channel_names'] = {v: k for k, v in channel_dict.items()}

    json_dict['labels'] = {
        "background": 0}
    if softseg:
        for i, contrast_class in enumerate(softseg):
            json_dict['labels'][f"{label_suffix}_{i + 1}"] = i + 1
    else:
        json_dict['labels'][label_suffix] = 1

    json_dict["numTraining"] = train_ctr + 1
    # Needed for finding the files correctly. IMPORTANT! File endings must match between images and segmentations!
    json_dict['file_ending'] = ".nii.gz"
    json_dict["overwrite_image_reader_writer"] = "SimpleITKIO"

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    main()
