## Using contrast agnostic weights with nnUNet

This guide provides instructions on how to train nnUNet models with the contrast-agnostic pre-trained weights. Ideal use cases include using the pretrained weights for training/finetuning on any spinal-cord-related segmentation task (e.g. lesions, rootlets, etc.). 

### Step 1: Download the pretrained weights

Download the pretrained weights from the [latest release](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases) of the contrast-agnostic model. This refers to the `.zip` file of the format `model_contrast_agnostic_<date>_nnunet_compatible.zip`. 

> [!WARNING]  
> Only download the model with the `nnunet_compatible` suffix. If a release does not have this suffix, then that model weights are not directly compatible with nnUNet.


### Step 2: Modify the plans file

In your  nnUNetPlans.json, change the `strides` key to the following: 
```
[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
```

Change `patch_size` for the `3D` model to: 
```
[64, 192, 320]
```

### Step 3: Train/Finetune the nnUNet model on your task

Provide the path to the downloaded pretrained weights:

```bash

nnUNetv2_train <dataset_name_or_id> <configuration> <fold> -pretrained_weights <path_to_pretrained_weights> -tr <nnUNetTrainer_Xepochs>

```

> [!IMPORTANT]  
> * Training/finetuning with contrast-agnostic weights only works when for 3D nnUNet models. 
> * Ensure that all images are in the RPI orientation before running nnUNetv2_plan_and_preprocess. This is because the updated `patch_size` refers to patches in RPI orientation (if images are in different orientation, then the patch size might be sub-optimal).
> * Ensure that `X` in `nnUNetTrainer_Xepochs` is set to a lower value than 1000. The idea is the finetuning does not require as many epochs as training from scratch because the contrast-agnostic model has already been trained on a lot of spinal cord images (so it might not require 1000 epochs to converge).

