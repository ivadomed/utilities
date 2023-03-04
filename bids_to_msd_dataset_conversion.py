"""
Template for converting a BIDS dataset to the Medical Segmentation Decathlon format. The ideal
use case for this template is to create a datalist JSON file, which can then be used with MONAI's 
`load_decathlon_datalist` function to create a PyTorch Dataset. Note that this template does NOT
restructure the existing BIDS dataset, but only creates a datalist JSON file, that points to the 
images and labels in the respective BIDS folders. 

Currently supports both single session (cross-sectional) and multi-session (longitudinal) BIDS datasets.

Some possible options of how the datalists can be created:
1. Single Session Single Contrast: Picks a single contrast and its corresponding GT label for each subject
2. Single Session Multi-Contrast: Within a single session, all contrasts or list of specified contrasts are picked 
    for each subject. Two options are provided for dealing with multiple contrasts:
    (i) Treat the contrasts independently to create separate (image, label) pairs for each contrast, OR,
    (ii) Group by contrasts and create a single (image, label) entity for each subject.
    NOTE: Currently assumes that all contrasts are co-registered hence only one GT label for all 
    the contrasts is used.
3. Multi-Session Single Contrast: From all sessions or a list of specified sessions, the corresponding (image, label) 
    pairs for a "single" contrast are picked. Two options are provided: 
    (i) Treat the sessions independently to create separate (image, label) pairs for each session, OR,
    (ii) Group by sessions and create a single (image, label) entity for each subject.
4. Multi-Session Multi-Contrast: This not currently supported and will soon be a feature

Some usage examples:
1. Single Session Single Contrast:
    python bids_to_msd_dataset_conversion.py --path-data /path/to/bids/dataset
      --path-out /path/to/output/directory --split 0.6 0.2 0.2 --label-suffix _lesion-manual --include-contrasts T2w

2. Single Session Multi-Contrast:
    Assuming 5 contrasts are available (T1w, T2w, FLAIR, PD, T2star) only choose FLAIR and T2w:
    python bids_to_msd_dataset_conversion.py --path-data /path/to/bids/dataset
        --path-out /path/to/output/directory --split 0.6 0.2 0.2 --include-contrasts FLAIR T2w --group-by-contrasts 
        --label-suffix _seg-lesion0 --common-label-contrast FLAIR

3. Multi-Session Single Contrast:
    Assuming 3 sessions are available (ses-01, ses-02, ses-03) only choose ses-01 and ses-03 and with sagittal images
    python bids_to_msd_dataset_conversion.py --path-data /path/to/bids/dataset
        --path-out /path/to/output/directory --split 0.6 0.2 0.2 --include-sessions ses-01 ses-03 --include-contrasts
        acq-sag_T2w --group-by-sessions
"""

# TODO: 
# - Add support for multi-session multi-contrast
# - Check if the (image, label) pairs exist before adding them to the datalist

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import utils_dataset_conversion as utils

parser = argparse.ArgumentParser(description='Code for creating datalists according to the MSD format.')

parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")
parser.add_argument('--path-data', default='./', required=True, type=str,
                    help='Absolute path to the data set directory')
parser.add_argument('--path-out', default='./', required=True, type=str,
                    help='Absolute path to the output directory where the datalist JSON will be saved.')
# argument that accepts a list of floats as train val test splits
parser.add_argument('--split', nargs='+', required=True, type=float, default=[0.6, 0.2, 0.2],
                    help='Ratios of training, validation and test splits lying between 0-1. Example: --split 0.6 0.2 '
                         '0.2')
# argument for getting the label suffix
parser.add_argument('--label-suffix', default='_lesion-manual', type=str, required=True,
                    help='Suffix for the label files.')
# argument that asks to group by sessions or not
parser.add_argument('--group-by-sessions', action='store_true',
                    help='Group images by sessions')
parser.add_argument('--group-by-contrasts', action='store_true',
                    help='Group images by contrasts')
parser.add_argument('--common-label-contrast', default='', type=str, required=False,
                    help='Common contrast (in proper BIDS suffix) whose label will be picked for all the other '
                         'contrasts (assuming co-registered). '
                         'Used only when --group-by-contrasts is set')
# argument that accepts a list of sessions to include in the dataset
parser.add_argument('--include-sessions', nargs='+', required=False, type=str, default=None,
                    help='Sessions (in proper BIDS suffixes) to include in the dataset. Note that only these sessions '
                         'will be picked to create the dataset. '
                         'If not used then all sessions will be included. Example: --include-sessions ses-01 ses-02')
# argument that accepts a list of contrasts to include in the dataset
parser.add_argument('--include-contrasts', nargs='+', required=False, type=str, default=None,
                    help='Contrasts (in proper BIDS suffixes) to include in the dataset. Note that only these sessions '
                         'will be picked to create the dataset. '
                         'If not used then all contrasts will be included. Example: --include-contrasts T1w T2w')

args = parser.parse_args()

root = args.path_data
train_ratio, val_ratio, test_ratio = args.split
PATH_DERIVATIVES = os.path.join(root, 'derivatives', 'labels')

# set the random number generator seed
rng = np.random.default_rng(args.seed)

# check if participants.tsv exists 
if not os.path.exists(os.path.join(root, 'participants.tsv')):
    raise FileNotFoundError("participants.tsv file not found in the dataset directory. Cannot proceed with subject "
                            "selection.")

# Get all subjects from participants.tsv
subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
subjects = subjects_df['participant_id'].values.tolist()
logger.info(f"Total number of subjects in the dataset: {len(subjects)}")

# NOTE: sklearn does not have function for training, validation and test split. 
# So, we use the following workaround to get the required splits
# Get only the training and test splits initially
train_subjects, test_subjects = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
# Use the training split to further split into training and validation splits
train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio / (train_ratio + val_ratio),
                                                random_state=args.seed, )

# boiler plate keys to be defined in the dataset.json
params = {}
params["description"] = "My awesome task"  # TODO: Add the name of the data
params["labels"] = {
    "0": "background",
    "1": "lesion/tumour/sc"  # TODO: Define the classes to be be segmented
}
params["license"] = "xxx"
params["modality"] = {
    "0": "MRI"
}
params["name"] = "dataset-name"  # TODO: Add the name of the dataset
params["reference"] = "N/A"
params["tensorImageSize"] = "3D"

subjects_dict = {
    "training": train_subjects,
    "validation": val_subjects,
    "test": test_subjects
}

# loop through the training, validation and test splits and create dictionaries that contain the 
# (image, label) pairs for each subject
for name, subs_list in subjects_dict.items():

    temp_list = []
    for subject_no, subject in enumerate(tqdm(subs_list, desc=f"Loading {name} volumes")):

        # recursively get all the files for the subject
        files = sorted(glob.glob(os.path.join(args.path_data, subject) + "/**/*.nii.gz", recursive=True))

        temp_data = {}
        session_ctr, contrast_ctr = 0, 0
        for file in files:
            # build file names
            subjectID, sessionID, contrast_suffixID, datatype, filename = utils.fetch_subject_info(file)
            # print(subjectID, sessionID, contrast_suffixID, datatype, filename)

            if sessionID == '':
                # print("No session ID found, possibly cross-sectional dataset. Moving on to check for grouping by contrasts....")

                if args.group_by_contrasts:
                    # print("Found that --group_by_contrasts is set. Moving on to check for common label suffix....")

                    # check if args.common_label is specified. If not, raise an error
                    error_msg = "Please specify the common label contrast using the --common_label_contrast " \
                                "argument. This will be used to pair all the contrasts with the same label file."
                    assert args.common_label_contrast != '', error_msg

                    # check if only the specified contrasts have to be included
                    if args.include_contrasts is not None and contrast_suffixID in args.include_contrasts:
                        image_file = os.path.join(root, subjectID, datatype, filename)

                        if contrast_suffixID == args.common_label_contrast:
                            label_file = os.path.join(PATH_DERIVATIVES, subjectID, datatype,
                                                      utils.add_suffix(filename, args.label_suffix))

                        # # TODO: check if there are missing contrasts

                    elif args.include_contrasts is None:
                        # assuming all contrasts have to be included
                        image_file = os.path.join(root, subjectID, datatype, filename)
                        if contrast_suffixID == args.common_label_contrast:
                            label_file = os.path.join(PATH_DERIVATIVES, subjectID, datatype,
                                                      utils.add_suffix(filename, args.label_suffix))

                    else:
                        print(f"Skipping contrasts {contrast_suffixID} for subject {subjectID} because not included")
                        continue

                    # Similar to nnUNet's naming convention, each contrast will be stored as image_0000, image_0001,
                    # etc. Since the label is common for all the contrasts, it will be stored as label_0000
                    temp_data[f"image_000{contrast_ctr}"] = image_file
                    temp_data[f"label_0000"] = label_file

                    # increment the contrast counter
                    contrast_ctr += 1

                else:
                    # if args.group_by_contrasts is not set, then each contrast will be stored as a separate image and label file
                    # print("Found that --group_by_contrasts is NOT set. Creating independent image and label files for each contrast ...")

                    temp_data = {}
                    # check if only the specified contrasts have to be included
                    if args.include_contrasts is not None and contrast_suffixID in args.include_contrasts:
                        image_file = os.path.join(root, subjectID, datatype, filename)
                        label_file = os.path.join(PATH_DERIVATIVES, subjectID, datatype,
                                                  utils.add_suffix(filename, args.label_suffix))

                    elif args.include_contrasts is None:
                        # assuming all contrasts have to be included
                        image_file = os.path.join(root, subjectID, datatype, filename)
                        label_file = os.path.join(PATH_DERIVATIVES, subjectID, datatype,
                                                  utils.add_suffix(filename, args.label_suffix))

                    else:
                        print(f"Skipping contrasts {contrast_suffixID} for subject {subjectID} because not included")
                        continue

                    # store in a temp dictionary
                    temp_data[f"image"] = os.path.join(root, subjectID, datatype, filename)
                    temp_data[f"label"] = os.path.join(PATH_DERIVATIVES, subjectID, datatype,
                                                       utils.add_suffix(filename, args.label_suffix))
                    temp_list.append(temp_data)

            else:
                # print("One or more sessions found. Moving on to check for grouping by sessions...")
                if args.group_by_sessions:
                    # print("Found that --group_by_sessions is set. Moving on to check if grouping by contrasts is set...")

                    if args.group_by_contrasts:
                        # print("Found that --group_by_contrasts is set. Moving on to check for common label suffix....")

                        print("Grouping by both sessions and contrasts is not currently supported. Grouping by "
                              "sessions only supports a single contrast")
                        exit()

                    else:
                        # print("Found that --group_by_contrasts is NOT set. Continuing to group by sessions only ...")

                        # check if only the specific sessions have to be included
                        if args.include_sessions is not None and sessionID in args.include_sessions and contrast_suffixID in args.include_contrasts:
                            image_file = os.path.join(root, subjectID, sessionID, datatype, filename)
                            label_file = os.path.join(PATH_DERIVATIVES, subjectID, sessionID, datatype,
                                                      utils.add_suffix(filename, args.label_suffix))

                        elif args.include_sessions is None and contrast_suffixID in args.include_contrasts:
                            # assuming all sessions have to be included
                            image_file = os.path.join(root, subjectID, sessionID, datatype, filename)
                            label_file = os.path.join(PATH_DERIVATIVES, subjectID, sessionID, datatype,
                                                      utils.add_suffix(filename, args.label_suffix))

                        else:
                            # if a session or contrast is not included, skip it
                            continue

                        # each session will be stored as image_01, image_01, etc. and label_01, label_02, etc.
                        temp_data[f"image_0{session_ctr + 1}"] = image_file
                        temp_data[f"label_0{session_ctr + 1}"] = label_file

                        # increment the session counter
                        session_ctr += 1

                else:
                    # print("Found that --group_by_sessions is NOT set. Using all sessions to create independent image and label files ...")
                    temp_data = {}

                    # check if only the specific sessions have to be included
                    if args.include_sessions is not None and sessionID in args.include_sessions and contrast_suffixID in args.include_contrasts:
                        image_file = os.path.join(root, subjectID, sessionID, datatype, filename)
                        label_file = os.path.join(PATH_DERIVATIVES, subjectID, sessionID, datatype,
                                                  utils.add_suffix(filename, args.label_suffix))

                    elif args.include_sessions is None and contrast_suffixID in args.include_contrasts:
                        # assuming all sessions have to be included
                        image_file = os.path.join(root, subjectID, sessionID, datatype, filename)
                        label_file = os.path.join(PATH_DERIVATIVES, subjectID, sessionID, datatype,
                                                  utils.add_suffix(filename, args.label_suffix))
                    else:
                        # if a session or contrast is not included, skip it
                        continue

                    # store in a temp dictionary
                    temp_data[f"image"] = image_file
                    # NOTE: Currently only works for single contrast (because label for that will be available)
                    temp_data[f"label"] = label_file
                    temp_list.append(temp_data)

        # only append the temp_list if grouping by sessions is set
        temp_list.append(temp_data) if args.group_by_sessions or args.group_by_contrasts else None
        # print(temp_list)

    params[name] = temp_list

# get the number of training, validation and test samples the depending on the grouping/sessions/contrasts chosen
params["numTraining"] = len(params["training"])
params["numValidation"] = len(params["validation"])
params["numTest"] = len(params["test"])

# check if output directory exists, if not create it
if not os.path.exists(args.path_out):
    os.makedirs(args.path_out)

final_json = json.dumps(params, indent=4, sort_keys=True)
jsonFile = open(os.path.join(args.path_out, "dataset.json"), "w")
jsonFile.write(final_json)
jsonFile.close()
print(f"{os.path.join(args.path_out, 'dataset.json')} file created successfully!")
