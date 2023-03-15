"""
Creates train/validation/test splits for a given dataset and saves the output in JSON file. 
The JSON file can be used to convert the dataset into nnUNet format using the script convert_bids_to_nnunet.py

Usage: 
1. If you want to include specific sessions and contrasts:
python create_data_splits.py --path-data <path-to-bids-dataset> --path-out <path-to-output-dir> --split 0.6 0.2 0.2 --include-sessions ses-01 ses-02 --include-contrasts T1w T2w

2. If you want to include all sessions for a specific contrast: (do not specify the --include-sessions argument)
python create_data_splits.py --path-data <path-to-bids-dataset> --path-out <path-to-output-dir> --split 0.6 0.2 0.2 --include-contrasts T1w T2w

Author: Naga Karthik
"""

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

parser = argparse.ArgumentParser(description='Code for creating data splits.')

parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")
parser.add_argument('--path-data', default='./', required=True, type=str,
                    help='Absolute path to the data set directory')
parser.add_argument('--path-out', default='./', required=True, type=str,
                    help='Absolute path to the output directory where the file *dataset.json* will be saved.')
# argument that accepts a list of floats as train val test splits
parser.add_argument('--split', nargs='+', required=True, type=float, default=[0.6, 0.2, 0.2],
                    help='Ratios of training, validation and test splits lying between 0-1. Example: --split 0.6 0.2 0.2')
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

subjects_dict = {
    "train": train_subjects,
    "valid": val_subjects,
    "test": test_subjects
}

# boiler plate keys to be defined in the dataset.json
params = {}

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

            if args.include_sessions is not None and sessionID in args.include_sessions and contrast_suffixID in args.include_contrasts:
                temp_list.append(filename)
            
            elif args.include_sessions is None and contrast_suffixID in args.include_contrasts:
                temp_list.append(filename)
            
            else:
                continue

        params[name] = temp_list

# get the number of training, validation and test samples the depending on the grouping/sessions/contrasts chosen
params["seed"] = args.seed
params["numTraining"] = len(params["train"])
params["numValidation"] = len(params["valid"])
params["numTest"] = len(params["test"])


# check if output directory exists, if not create it
if not os.path.exists(args.path_out):
    os.makedirs(args.path_out)

final_json = json.dumps(params, indent=4, sort_keys=True)
jsonFile = open(os.path.join(args.path_out, "split-dataset.json"), "w")
jsonFile.write(final_json)
jsonFile.close()
print(f"{os.path.join(args.path_out, 'split-dataset.json')} file created successfully!")
print(f"To use the output JSON file for converting a BIDS dataset into the nnUNet format use the following command: "  
        "\npython convert_bids_to_nnunet.py --path-data <path-to-bids-dataset> --path-out <path-to-output-dir> --split-dict <path-to/split-dataset.json> --taskname <task-name> --tasknumber <task-number>\n")
