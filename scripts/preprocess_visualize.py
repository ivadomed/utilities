"""
This script is to visualize a preprocessed image based on the plans.json file which is self configured by nnUNet.

Usage:
python preprocess_visualize.py HOME_FOLDER/nnUNetTrainer__nnUNetPlans__{CONFIG}/plans.json PATH_TO_INPUT_IMAGE.nii.gz PATH_TO_OUTPUT_FOLDER NNUNET_CONFIG

Expected output: This script will create an output folder which would have all the patches that would be extracted from the input image for the training.

Authors: Rohan Banerjee
"""

import os
import json
import argparse
import SimpleITK as sitk
import numpy as np

def extract_patches(input_image, patch_size):
    input_np = sitk.GetArrayFromImage(input_image)
    depth, height, width = input_np.shape
    patches = []

    for i in range(0, depth, patch_size[2]):
        for j in range(0, height, patch_size[1]):
            for k in range(0, width, patch_size[0]):
                patch = input_np[i:i+patch_size[2], j:j+patch_size[1], k:k+patch_size[0]]
                patches.append(patch)

    return patches

def preprocess_patch(patch, preprocessing_plan):

    config_3d_fullres = preprocessing_plan['configurations']['3d_fullres']

    mean_intensity = np.mean(patch)
    std_intensity = np.std(patch)

    z_score_patch = (patch - mean_intensity) / std_intensity

    return z_score_patch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NIFTI image with a nnUNet plans.json file")
    parser.add_argument("json_file", help="Path to the plans.json configuration file")
    parser.add_argument("input_nifti", help="Path to the input NIFTI image")
    parser.add_argument("output_folder", help="Path to the output folder for saving patches")
    parser.add_argument("config", help="nnUNet Training configuration. Eg: 2d, 3d_fullres etc")

    args = parser.parse_args()

    with open(args.json_file, 'r') as plan_file:
        preprocessing_plan = json.load(plan_file)

    input_image = sitk.ReadImage(args.input_nifti)
    patch_size = preprocessing_plan['configurations'][args.config]['patch_size']

    patches = extract_patches(input_image, patch_size)
    for i, patch in enumerate(patches):
        z_score_patch = preprocess_patch(patch, preprocessing_plan)

        patch_output_path = os.path.join(args.output_folder, f"patch_{i}.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(z_score_patch), patch_output_path)

    print("Patches saved in", args.output_folder)