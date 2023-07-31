#!/bin/bash
#
# Training nnUNetv2 on multiple folds
#
# NOTE: This is a template script, modify it as needed
#
# Authors: Naga Karthik, Jan Valosek
#

echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"
nnUNetv2_plan_and_preprocess -d <DATASET_ID> --verify_dataset_integrity -c <CONFIG>     # <CONFIG> is optional, might be 2d, 3d_fullres, 3d_lowres, or 3d_cascade_fullres

# Select number of folds here
# folds=(0 1 2 3 4)
# folds=(0 1 2)
folds=(0)

for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"

    # training
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_train <DATASET_ID> <CONFIG> $fold

    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

    # inference
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/<DATASET_ID>/imagesTs -o <OUT_DIR>/nnUNetPlans__<CONFIG>/fold_$fold/test -d <DATASET_ID> -f $fold -c <CONFIG>

    echo ""
    echo "-------------------------------------------"
    echo " Inference completed on Fold $fold"
    echo "-------------------------------------------"

done
