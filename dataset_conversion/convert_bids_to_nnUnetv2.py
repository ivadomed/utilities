"""
Converts BIDS-structured dataset to the nnUNetv2 dataset format. Full details about
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Note that the conversion from BIDS to nnUNet is done using symbolic links to avoid creating multiple copies of the 
(original) BIDS dataset.

Currently only supports the conversion of a single contrast. In case of multiple contrasts, the script should be 
modified to include those as well. 

Usage example:
    python convert_bids_to_nnUNetv2.py --path-data ~/data/dataset --path-out ~/data/dataset-nnunet
                    --dataset-name MyDataset --dataset-number 501 --split 0.6 0.2 0.2 --seed 99

Naga Karthik, Jan Valosek modified by Théo Mathieu
"""

import argparse
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
    parser.add_argument('--path-data', required=True,
                        help='Path to BIDS dataset. Example: ~/data/dataset')
    parser.add_argument('--path-out', required=True,
                        help='Path to output directory. Example: ~/data/dataset-nnunet')
    parser.add_argument('--contrast', required=True, type=str,
                        help='Subject contrast. Example: T2w or acq-sag_T2w')
    parser.add_argument('--label-suffix', required=True, type=str,
                        help='Label suffix. Example: lesion-manual or seg-manual')
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

    return parser


def binarize_label(subject_path, label_path):
    label_npy = nib.load(label_path).get_fdata()
    # NOTE: using a very small threshold (<<< 0) to binarize the label leads to more 
    # more volume of the label being retained. For e.g. due to PVE, the voxels which have 
    # value of 0.0001 in the label file will still be retained in the binarized label as 1.
    # Since this is not a correct representation of the label, we use a threshold of 0.5.
    threshold = 0.5
    label_npy = np.where(label_npy > threshold, 1, 0)
    ref = nib.load(subject_path)
    label_bin = nib.Nifti1Image(label_npy, ref.affine, ref.header)
    # overwrite the original label file with the binarized version
    nib.save(label_bin, label_path)


def main():
    parser = get_parser()
    args = parser.parse_args()

    root = Path(os.path.abspath(os.path.expanduser(args.path_data)))
    path_out = Path(os.path.join(os.path.abspath(os.path.expanduser(args.path_out)),
                                 f'Dataset{args.dataset_number:03d}_{args.dataset_name}'))

    # Get filename
    contrast = args.contrast
    label_suffix = args.label_suffix

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
    train_subjects, test_subjects = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
    rng.shuffle(train_subjects)

    # Initialize counters for train and test subjects
    train_ctr, test_ctr = 0, 0
    # Loop through all subjects
    for subject in subjects:

        # Train subjects
        if subject in train_subjects:

            # Session folder(s) exist
            # Check if session folder(s) exist
            if any('ses' in folder for folder in os.listdir(os.path.join(root, subject))):
                # Get all sessions for the subject
                sessions = os.listdir(os.path.join(root, subject))
                sessions.sort()
                # Get number of sessions for the subject
                num_sessions_per_subject = len(os.listdir(os.path.join(root, subject)))

                for session in sessions:
                    train_ctr += 1

                    subject_image_file = os.path.join(root, subject, session, 'anat',
                                                      f"{subject}_{session}_{contrast}.nii.gz")
                    subject_label_file = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat',
                                                      f"{subject}_{session}_{contrast}_{label_suffix}.nii.gz")

                    # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding
                    # nnunet convention names

                    # create the new convention names for nnunet
                    sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + \
                                   str(Path(subject_image_file).name).split('_')[1]
                    subject_image_file_nnunet = os.path.join(path_out_imagesTr,
                                                             f"{sub_ses_name}_{train_ctr:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTr,
                                                             f"{sub_ses_name}_{train_ctr:03d}.nii.gz")

                    train_images.append(subject_image_file_nnunet)
                    train_labels.append(subject_label_file_nnunet)

                    # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                    os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                    os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                    # binarize the label file
                    #binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

            # No session folder(s) exist
            # TODO: there is a lot of code duplication with the above if statement --> refactor
            else:
                train_ctr += 1
                subject_image_file = os.path.join(root, subject, 'anat',
                                                  f"{subject}_{contrast}.nii.gz")
                subject_label_file = os.path.join(root, 'derivatives', 'labels', subject, 'anat',
                                                  f"{subject}_{contrast}_{label_suffix}.nii.gz")

                # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding
                # nnunet convention names

                # create the new convention names for nnunet
                sub_name = str(Path(subject_image_file).name).split('_')[0]
                subject_image_file_nnunet = os.path.join(path_out_imagesTr,
                                                         f"{sub_name}_{train_ctr:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTr,
                                                         f"{sub_name}_{train_ctr:03d}.nii.gz")

                train_images.append(subject_image_file_nnunet)
                train_labels.append(subject_label_file_nnunet)

                # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                # binarize the label file
                #binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

        # Test subjects
        # TODO: there is a lot of code duplication with the train subjects loop --> refactor
        elif subject in test_subjects:

            # Session folder(s) exist
            # Check if session folder(s) exist
            if any('ses' in folder for folder in os.listdir(os.path.join(root, subject))):
                # Get all sessions for the subject
                sessions = os.listdir(os.path.join(root, subject))
                sessions.sort()
                # Get number of sessions for the subject
                num_sessions_per_subject = len(os.listdir(os.path.join(root, subject)))

                for session in sessions:
                    test_ctr += 1

                    subject_image_file = os.path.join(root, subject, session, 'anat',
                                                      f"{subject}_{session}_{contrast}.nii.gz")
                    subject_label_file = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat',
                                                      f"{subject}_{session}_{contrast}_{label_suffix}.nii.gz")

                    # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding
                    # nnunet convention names

                    # create the new convention names for nnunet
                    sub_ses_name = str(Path(subject_image_file).name).split('_')[0] + '_' + \
                                   str(Path(subject_image_file).name).split('_')[1]
                    subject_image_file_nnunet = os.path.join(path_out_imagesTs,
                                                             f"{sub_ses_name}_{train_ctr:03d}_0000.nii.gz")
                    subject_label_file_nnunet = os.path.join(path_out_labelsTs,
                                                             f"{sub_ses_name}_{train_ctr:03d}.nii.gz")

                    train_images.append(subject_image_file_nnunet)
                    train_labels.append(subject_label_file_nnunet)

                    # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                    os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                    os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                    # binarize the label file
                    #binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

            # No session folder(s) exist
            else:
                test_ctr += 1
                subject_image_file = os.path.join(root, subject, 'anat',
                                                  f"{subject}_{contrast}.nii.gz")
                subject_label_file = os.path.join(root, 'derivatives', 'labels', subject, 'anat',
                                                  f"{subject}_{contrast}_{label_suffix}.nii.gz")

                # NOTE: if adding more contrasts, add them here by creating image-label files and the corresponding
                # nnunet convention names

                # create the new convention names for nnunet
                sub_name = str(Path(subject_image_file).name).split('_')[0]
                #TODO modify channel to adapt on contrast
                subject_image_file_nnunet = os.path.join(path_out_imagesTs,
                                                         f"{sub_name}_{train_ctr:03d}_0000.nii.gz")
                subject_label_file_nnunet = os.path.join(path_out_labelsTs,
                                                         f"{sub_name}_{train_ctr:03d}.nii.gz")

                train_images.append(subject_image_file_nnunet)
                train_labels.append(subject_label_file_nnunet)

                # copy the files to new structure using symbolic links (prevents duplication of data and saves space)
                os.symlink(os.path.abspath(subject_image_file), subject_image_file_nnunet)
                os.symlink(os.path.abspath(subject_label_file), subject_label_file_nnunet)

                # binarize the label file
                #binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

        else:
            print("Skipping file, could not be located in the Train or Test splits split.", subject)

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
    json_dict['name'] = args.dataset_name
    json_dict['description'] = args.dataset_name
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"
    json_dict['numTraining'] = train_ctr
    json_dict['numTest'] = test_ctr

    # The following keys are the most important ones.
    #TODO adapt to config V2
    """
    channel_names:
        Channel names must map the index to the name of the channel. For BIDS, this refers to the contrast suffix.
        {
            0: 'T1',
            1: 'CT'
        }
    Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }
        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!
    """

    json_dict['channel_names'] = {
        0: "acq-sag_T2w",
    }

    json_dict['labels'] = {
        "background": 0,
        "lesion": 1,
    }

    # Needed for finding the files correctly. IMPORTANT! File endings must match between images and segmentations!
    json_dict['file_ending'] = ".nii.gz"

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    main()