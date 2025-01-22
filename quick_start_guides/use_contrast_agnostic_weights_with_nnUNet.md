## Using contrast agnostic weights with nnUNet

This guide provides instructions on how to train nnUNet models with the contrast-agnostic pre-trained weights. Ideal use cases include using the 
pretrained weights for training/finetuning on any spinal-cord-related segmentation task (e.g. lesions, rootlets, etc.). 

### Step 1: Download the pretrained weights

Download the pretrained weights from the [latest release](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases) of the 
contrast-agnostic model. This refers to the `.zip` file of the format `model_contrast_agnostic_<date>_nnunet_compatible.zip`. 

> [!WARNING]  
> Only download the model with the `nnunet_compatible` suffix. If a release does not have this suffix, then that model weights are not directly 
compatible with nnUNet.


### Step 2: Create a new plans file

1. Create a copy of the original `nnUNetPlans.json` file (found under `$nnUNet_preprocessed/<dataset_name_or_id>`) and 
rename it to `nnUNetPlans_contrast_agnostic.json`. 
2. In the `nnUNetPlans_contrast_agnostic.json`, modify the values of the following keys in the `3d_fullres` dict to be able to match the 
values used for the contrast-agnostic model:

```json

"patch_size": [64, 192, 320],
"n_stages": 6,
"features_per_stage": [32, 64, 128, 256, 320, 320],
"n_conv_per_stage": [2, 2, 2, 2, 2, 2],
"n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
"strides": [
    [1, 1, 1],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [1, 2, 2]
    ],
"kernel_sizes":[
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3]
    ],
"strides": [
    [1, 1, 1],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [1, 2, 2]
    ],

```

### Step 3: Train/Finetune the nnUNet model on your task

Provide the path to the downloaded pretrained weights:

```bash

nnUNetv2_train <dataset_name_or_id> <configuration> <fold> -p nnUNetPlans_contrast_agnostic.json -pretrained_weights <path_to_pretrained_weights> -tr <nnUNetTrainer_Xepochs>

```

> [!IMPORTANT]  
> * Training/finetuning with contrast-agnostic weights only works when for 3D nnUNet models. 
> * Ensure that all images are in the RPI orientation before running `nnUNetv2_plan_and_preprocess`. This is because the updated 
`patch_size` refers to patches in RPI orientation (if images are in different orientation, then the patch size might be sub-optimal).
> * Ensure that `X` in `nnUNetTrainer_Xepochs` is set to a lower value than 1000. The idea is the finetuning does not require as 
many epochs as training from scratch because the contrast-agnostic model has already been trained on a lot of spinal cord images 
(so it might not require 1000 epochs to converge).
> * The modified `nnUNetPlans_contrast_agnostic.json` might not have the same values for parameters such as `patch_size`, `strides`, 
`n_stages`, etc. automatically set by nnUNet. As a result, the original (train-from-scratch) model might even perform better. 
In such cases, training/finetuning with contrast-agnostic weights should just be considered as another baseline for your actual task.

