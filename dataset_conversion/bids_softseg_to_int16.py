import argparse
import os

import nibabel as nib
import numpy as np
import pathlib


# TODO MORE TEST
def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert softseg float value to integer class value')
    parser.add_argument('--path-data', required=True, help='Path to BIDS dataset. Example: ~/data/dataset')
    parser.add_argument('--path-out', required=True, help='Path to save img')
    parser.add_argument('--softseg', nargs='+', type=float, help='Voxel value class name (separated with space).'
                                                                 'If the label file is a soft segmentation, voxel value'
                                                                 ' will be discretize in class. Example:'
                                                                 '--softseg 0.001 0.25 0.5 0.75  '
                                                                 '(4 class with values [0.001, 0.25), [0.25, 0.5), '
                                                                 '[0.5, 0.75), [0.75, 1) respectively')
    return parser


def discretise_soft_seg(label_file, interval, out):
    """
    Discretize softseg in integer class.
    Args:
        label_file (str): Path to the label file
        interval (list): List with class boundary
    """
    nifti_file = nib.load(label_file)
    data = nifti_file.get_fdata()
    class_voxel = np.zeros_like(data, dtype=np.int16)
    for i, value in enumerate(interval):
        lower_bound = interval[i]
        if i == len(interval) - 1:
            class_voxel[data >= lower_bound] = i + 1
        else:
            upper_bound = interval[i + 1]
            class_voxel[(data >= lower_bound) & (data < upper_bound)] = i + 1
    voxel_img = nib.Nifti1Image(class_voxel, nifti_file.affine, nifti_file.header)
    nib.save(voxel_img, out)


def main():
    parser = get_parser()
    args = parser.parse_args()
    path_in = args.path_data
    path_out = args.path_out
    softseg = args.softseg
    if softseg:
        for i in range(1, len(softseg)):
            if softseg[i] <= softseg[i - 1]:
                raise ValueError(f"Softseg values {softseg} are not increasing.")
    print(path_in, softseg)
    for sub in os.listdir(path_in):
        if sub.startswith('sub-'):
            for image in os.listdir(os.path.join(path_in, sub, "anat")):
                if not image.endswith(".json"):
                    pathlib.Path(path_out, sub, "anat").mkdir(parents=True, exist_ok=True)
                    discretise_soft_seg(os.path.join(path_in, sub, "anat", image), softseg,
                                        os.path.join(path_out, sub, "anat", image))


if __name__ == '__main__':
    main()
