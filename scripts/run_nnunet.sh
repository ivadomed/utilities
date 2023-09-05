#!/bin/bash
#
# Training nnUNetv2 on multiple folds
#
# NOTE: This is a template script, modify it as needed
#
# Authors: Naga Karthik, Jan Valosek
#

# !!! MODIFY THE FOLLOWING VARIABLES ACCORDING TO YOUR NEEDS !!!
config=CONFIG                     # e.g. 3d_fullres or 2d
dataset_id=DATASET_ID             # e.g. 301
dataset_name=DATASET_NAME         # e.g Dataset301_XXX
nnunet_trainer="nnUNetTrainer"

# Select number of folds here
# folds=(0 1 2 3 4)
# folds=(0 1 2)
folds=(0)

echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"

nnUNetv2_plan_and_preprocess -d ${dataset_id} --verify_dataset_integrity

for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"

    # training
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_train ${dataset_id} ${config} ${fold} -tr ${nnunet_trainer}

    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

    # inference
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs -tr ${nnunet_trainer} -o ${nnUNet_results}/${nnunet_trainer}__nnUNetPlans__${config}/fold_${fold}/test -d ${dataset_id} -f ${fold} -c ${config}

    echo ""
    echo "-------------------------------------------"
    echo " Inference completed on Fold $fold"
    echo "-------------------------------------------"

done
