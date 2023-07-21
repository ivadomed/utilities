"""
Concat 2 nnUNet dataset into one.

Usage example:
    python concat_nnUnet_dataset --path-in ~/Downloads/Dataset_002 ~/Downloads/Dataset_004 --path_out Dataset_new --copy

Théo Mathieu
"""
# TODO check compatible orientation
import argparse
import shutil
import os
import pandas as pd
import json

class CustomException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--path-in', required=True, nargs='+',
                        help='Path to nnUNet dataset. Example: ~/data/dataset')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-nnunet')
    parser.add_argument('--copy', '-cp', type=bool, default=False,
                        help='Making symlink (False) or copying (True) the files in the new dataset, '
                             'default = False. Example for symlink: --copy True')
    return parser

def compare_list_of_dicts(dict_list):
    if len(dict_list) < 2:
        return True

    keys_set = set(dict_list[0].keys())
    for dictionary in dict_list[1:]:
        if set(dictionary.keys()) != keys_set or dictionary != dict_list[0]:
            return False

    return True

def compare_list(list_2_comp):
    return all(element == list_2_comp[0] for element in list_2_comp)

def main():
    parser = get_parser()
    args = parser.parse_args()
    path_in = args.path_in
    path_out = args.path_out
    copy = args.copy
    all_info = {}

    for dataset in path_in:
        all_info[dataset] = json.load(open(os.path.join(dataset,"dataset.json")))

    same_label = compare_list_of_dicts([all_info[d]["labels"] for d in all_info.keys()])
    if not same_label:
        raise CustomException(f"""At least 2 of your dataset don't have the same labels specificity, {[f"{d} : {all_info[d]['labels']}" for d in all_info.keys()]}""")
    same_file_end = compare_list([all_info[d]["file_ending"] for d in all_info.keys()])


if __name__ == '__main__':
    main()
