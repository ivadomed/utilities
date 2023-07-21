"""
Extract image from BIDS dataset into one folder compatible with nnUNet inference.
More information about nnUNet inference format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format_inference.md

Usage example:
    python extract_bids_subject.py --path-bids ~/data/dataset --path-out ~/data/dataset-nnunet --contrast T2w --suffix 0000

Théo Mathieu
"""
import argparse
import os
import shutil
import pandas as pd


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Extract BIDS image into a folder compatible with nnUNet inference.')
    parser.add_argument('--path-bids', required=True, help='Path to BIDS dataset. Example: ~/data/dataset')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-nnunet')
    parser.add_argument('--contrast', required=True, type=str, help='Contrast of the images to extract. Example: T2w')
    parser.add_argument('--suffix', required=True, type=int,
                        help='Contrast in the nnUNet format, 4 digit (maximum) corresponding to the one use in your dataset.json for training. Example: 0')
    parser.add_argument('--copy', '-cp', type=bool, default=False,
                        help='Making symlink (False) or copying (True) the files in the nnUNet dataset, '
                             'default = False. Example for symlink: --copy True')
    parser.add_argument('--log', type=bool, default=False,
                        help='Save a csv files with path before and after extraction of the files. Default = false')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    copy = args.copy
    bids_path = args.path_bids
    out_path = args.path_out
    contrast = args.contrast
    suffix = args.suffix
    log = args.log
    extracted_files = {"old_path":[], "new_path":[], "copy/symlink":[]}
    for item in os.listdir(bids_path):
        if os.path.isdir(os.path.join(bids_path, item)) and item.startswith("sub-"):
            for root, _, list_files in os.walk(os.path.join(bids_path,item)):
                for file in list_files:
                    if file.endswith(f"_{contrast}.nii.gz"):
                        extracted_files["old_path"].append(os.path.join(bids_path, item, root, file))
    for i,old_path in enumerate(extracted_files["old_path"]):
        old_name = old_path.split('/')[-1].split('_')[0]
        new_path = os.path.join(out_path, f"{old_name}_{i:03}_{suffix:04}.nii.gz")
        if copy:
            shutil.copyfile(old_path, new_path)
            extracted_files["new_path"].append(new_path)
            extracted_files["copy/symlink"].append("copy")
        else:
            os.symlink(old_path, new_path)
            extracted_files["new_path"].append(new_path)
            extracted_files["copy/symlink"].append("symlink")
    if log:
        df = pd.DataFrame(extracted_files)
        df.to_csv(os.path.join(out_path, f"extracted_file.csv"), index=False)


if __name__ == '__main__':
    main()