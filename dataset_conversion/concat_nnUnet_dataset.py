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
import pathlib


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
                        help='Making symlink (False) or copying (True) the files in the nnUNet dataset, '
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


def change_cannel_suffix(new_val):
    print(new_val)


def main():
    parser = get_parser()
    args = parser.parse_args()
    path_in = args.path_in
    path_out = args.path_out
    copy = args.copy
    all_info = {}
    for dataset in path_in:
        all_info[dataset] = json.load(open(os.path.join(dataset, "dataset.json")))
    same_label = compare_list_of_dicts([all_info[d]["labels"] for d in all_info.keys()])
    if not same_label:
        raise CustomException(
            f"""At least 2 of your dataset don't have the same labels specificity, {[f"{d} : {all_info[d]['labels']}" for d in all_info.keys()]}""")
    same_file_end = compare_list([all_info[d]["file_ending"] for d in all_info.keys()])
    if not same_file_end:
        raise CustomException(
            f"""At least 2 of your dataset have not the same file ending, 
            {[f"{d} : {all_info[d]['file_ending']}" for d in all_info.keys()]}""")

    new_channel = {}
    for dataset in all_info.keys():
        original = all_info[dataset]["channel_names"]
        for old_key in original.keys():
            if old_key in new_channel:
                if original[old_key] == new_channel[old_key]:
                    print("same")
                # TODO add support to convert channel if not the same
                else:
                    raise CustomException(
                        f"""At least 2 of your dataset have incompatible channel name, 
                        {[f"{d} : {all_info[d]['channel_names']}" for d in all_info.keys()]}""")
            else:
                new_channel[old_key] = original[old_key]

    csv_log = {"old_name": [], "new_name":[], "old_dataset":[], "new_dataset":[]}
    labels = ["labelsTr", "labelsTs"]
    id_nb = 0
    for dataset in path_in:
        for i, folder in enumerate(["imagesTr", "imagesTs"]):
            pathlib.Path(os.path.join(path_out, folder)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.join(path_out, labels[i])).mkdir(parents=True, exist_ok=True)
            for old_name in os.listdir(os.path.join(dataset, folder)):
                if old_name.startswith("sub-"):
                    # image name
                    csv_log["old_name"].append(old_name)
                    csv_log["old_dataset"].append(os.path.join(dataset, folder))
                    channel = old_name.split('_')[-1]
                    base_name = '_'.join(old_name.split('_')[:-2])
                    new_name = f"{base_name}_{id_nb:03d}_{channel}"
                    csv_log["new_name"].append(new_name)
                    csv_log["new_dataset"].append(os.path.join(path_out, folder))

                    # label name
                    old_label = f"{'_'.join(old_name.split('_')[:-1])}.nii.gz"
                    old_label_full = os.path.join(dataset, labels[i], old_label)
                    new_label = f"{base_name}_{id_nb:03d}.nii.gz"
                    csv_log["old_name"].append(old_label)
                    csv_log["old_dataset"].append(os.path.join(dataset, labels[i]))
                    csv_log["new_name"].append(new_label)
                    csv_log["new_dataset"].append(os.path.join(path_out, labels[i]))
                    if copy:
                        shutil.copy(os.path.join(dataset, folder, old_name), os.path.join(path_out, folder, new_name))
                        shutil.copy(old_label_full, os.path.join(path_out, labels[i], new_label))
                    else:
                        os.symlink(os.path.join(dataset, folder, old_name), os.path.join(path_out, folder, new_name))
                        os.symlink(old_label_full, os.path.join(path_out, labels[i], new_label))
                    id_nb += 1



    df = pd.DataFrame(csv_log)
    df.to_csv(os.path.join(path_out, "log.csv"), index=False)






if __name__ == '__main__':
    main()
