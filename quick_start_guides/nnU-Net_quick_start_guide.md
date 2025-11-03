# nnU-Net quick-start guide

This file provides a quick-start guide for nnU-Net v2.

nnU-Net is a self-configuring framework for deep learning-based medical image segmentation; see [nnUNet GitHub page](https://github.com/MIC-DKFZ/nnUNet) and [publication](https://www.nature.com/articles/s41592-020-01008-z).

## Table of Contents

0. [Using Neuropoly's fork of nnU-Net](#0-using-neuropolys-fork-of-nnu-net)
1. [Installation](#1-installation)
   1. [`git clone` + Python virtual environment](#i-git-clone)
   2. [`conda` environment](#ii-conda)
   3. [Upgrading `nnunetv2`](#iii-upgrading-nnunetv2)
2. [Setting required environment variables](#2-environment-variables)
3. [Data structure](#3-data-structure)
4. [Train a model](#4-train-a-model)
   1. [Validate dataset integrity](#i-validate-dataset-integrity)
   2. [Using a custom trainer](#ii-using-a-custom-trainer)
   3. [Run training](#iii-run-training)
5. [Run prediction/inference](#5-run-predictioninference)
6. [Compute segmentation metrics](#6-compute-segmentation-metrics)

----

## 0. Using Neuropoly's fork of nnU-Net

> [!IMPORTANT]
> 
> In October 2025, NeuroPoly "forked" the nnU-Netv2 repo (`nnunetv2` -> [`nnunetv2-neurpoly`](https://github.com/spinalcordtoolbox/nnUNet-neuropoly)). This fork provides us with some extra freedom to tweak the nnU-Net package to meet our needs.
> 
> Some of the benefits of this fork include:
> 
> - Tested compatibility with SCT.
> - Compatibility with older versions of PyTorch.
> - Improved support for custom trainer classes.
> - Improved support for multi-fold inference.
> 
> All of the links and instructions in this document will point to the fork.
>
> Outside of these instructions, you can generally replace "`pip install nnunetv2`" with "`pip install nnunetv2-neuropoly`" and it will work exactly the same as `nnunetv2`.

----

## 1. Installation

Official installation instructions are available [here](https://github.com/spinalcordtoolbox/nnUNet-neuropoly/blob/master/documentation/installation_instructions.md).

> **Note**
> Always install nnU-Net inside a virtual environment.

> **Note**
> Run the installation commands on a GPU cluster, not on your laptop.

---

### i. `git clone`

`python -m venv` and `git clone`:

```bash
# Create and activate a Python virtual environment
cd ~
mkdir nnUNet_env
python -m venv nnUNet_env/
source nnUNet_env/bin/activate
# Clone the repository
git clone -b neuropoly-fork-patches git@github.com:spinalcordtoolbox/nnUNet-neuropoly.git
# Open the repostiroy folder and install
cd nnUNet
pip install -e .
```

---

### ii. `conda`

```bash
# create conda env
conda create --name nnunet python=3.10
conda activate nnunet
```

**GPU `conda install`:**

```bash
# install pytorch using conda - https://pytorch.org/get-started/locally/
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
# install nnunet (neuropoly fork)
pip install nnunetv2-neuropoly
# Install hiddenlayer. hiddenlayer enables nnU-net to generate plots of the network topologies it generates
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

**GPU `pip3 install`:**

```bash
# install pytorch using pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2-neuropoly
# Install hiddenlayer. hiddenlayer enables nnU-net to generate plots of the network topologies it generates
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

To verify that your `pytorch` installation supports CUDA, start `python` and run the following commands:

```python
import torch
print(torch.cuda.is_available())
```

This should now return `True`.

ℹ️ If you encounter issues during installation, please report them to [this issue](https://github.com/ivadomed/utilities/issues/45).

**CPU (for inference only):**

```bash
# install pytorch - https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# install nnunet (neuropoly fork)
pip install nnunetv2-neuropoly
# Install hiddenlayer. hiddenlayer enables nnU-net to generate plots of the network topologies it generates
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

---

### iii. Upgrading nnunetv2

To upgrade nnunetv2 to the latest version, you can run the following command in your virtual env:

```bash
pip install --upgrade nnunetv2-neuropoly
```

To check the current version and to upgrade to a specific version, you can use: 

```bash
pip freeze | grep nnunet-neuropoly
pip install nnunetv2-neuropoly==2.6.2
```

> [!WARNING]
> The NeuroPoly fork currently supports version `2.6.2` and above.
> 
> If you need an older version, please open an issue on the SCT repo explaining your use-case, and SCT's devs will create a release for that older version.

----

## 2. Environment variables

For details, see [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md#linux--macos).

nnU-Net requires the following three directories: `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`. You can create them using the commands below.

> **Note**
> Typically, these folders need to be created on the GPU server, not on the computer. You can connect to our GPU servers using `ssh`; see the intranet for details.

```bash
cd ~
mkdir data/nnunetv2
cd data/nnunetv2
mkdir nnUNet_raw nnUNet_preprocessed nnUNet_results
```

Then, include variables with paths to these folders in your `~/.bashrc` or `~/.zshrc` file:

```
export nnUNet_raw="${HOME}/data/nnunetv2/nnUNet_raw"
export nnUNet_preprocessed="${HOME}/data/nnunetv2/nnUNet_preprocessed"
export nnUNet_results="${HOME}/data/nnunetv2/nnUNet_results"
```

> **Note**
> Modify the paths according to where you created the folders.

## 3. Data structure

nnU-Net expects the following data structure (see [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#dataset-folder-structure)) for details):

```
nnUNet_raw/Dataset001_NAME1
├── dataset.json
├── imagesTr
│   ├── sub-amu01_T2w_001_0000.nii.gz        # The last 4-digit (`0000`) are used to denote channels; if you have more channels (or MRI contrasts, e.g., T1w, T2w), use  `0000`, `0001`, `0002`, etc.
│   ├── sub-amu02_T2w_002_0000.nii.gz
│   ├── ...
├── imagesTs
│   ├── sub-mgh01_T2w_089_0000.nii.gz
│   ├── sub-mgh02_T2w_090_0000.nii.gz
│   ├── ...
└── labelsTr
    ├── sub-amu01_T2w_001.nii.gz
    ├── sub-amu02_T2w_002.nii.gz
    ├── ...
```

- **imagesTr** contains the images belonging to the training cases. nnU-Net will perform pipeline configuration, training with 
cross-validation, as well as finding postprocessing and the best ensemble using this data. 
- **imagesTs** (optional) contains the images that belong to the test cases. nnU-Net does not use them! This could just 
be a convenient location for you to store these testing images.
- **labelsTr** contains the images with the ground truth segmentation labels for the training cases.
- **labelsTs** (optional) contains the images with the ground truth segmentation labels for the testing cases. 
- **dataset.json** contains metadata of the dataset (more details [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson)). Example:

```json
{ 
 "channel_names": {
   "0": "T2w"
 }, 
 "labels": {
   "background": 0,
   "sc_seg": 1
 }, 
 "numTraining": 32, 
 "file_ending": ".nii.gz"
 "overwrite_image_reader_writer": "SimpleITKIO"
 }
```
  
You can use [our scripts](https://github.com/ivadomed/utilities/tree/main/dataset_conversion) to convert the data from BIDS to the nnU-Net format. 

> **Note**
> It is a good idea to reorient all the images into a common orientation (e.g., `RPI`) before running training.
> TODO: list some of our previous discussions
> TODO: mention also resampling into common resolution? 

----

## 4. Train a model

> **Note**
> Since you will likely be running the training on one of our GPU servers, you will need to get your training data there. See [our intranet](https://intranet.neuro.polymtl.ca/computing-resources/neuropoly/gpus.html#data) for details. 

> **Note**
> Always run training inside the virtual terminal. You can use [`screen`](https://intranet.neuro.polymtl.ca/geek-tips/bash-shell/README.html#screen-for-background-processes) or `tmux`.

### i. Validate dataset integrity.

> Note that if you only plan to use 2d, 3d_fullres or 3d_lowres data, you should use the flag `-c <DATA_TYPE>` to only generate the wanted data and save some space! (default: -c 2d 3d_fullres 3d_lowres)

> Also 3d_cascade_fullres uses 3d_fullres data

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -c 2d 3d_fullres 3d_lowres
```

Replace `DATASET_ID` with a number higher than 500, for example, `-d 501`.

### ii. Using a custom trainer

Because we are using the NeuroPoly fork of nnUNet, you can create and specify a custom trainer class if the default trainers aren't sufficient for your needs. 

Steps:

- Create a new Python file called `trainer_class.py` using the following template:
   ```python
   import torch
   
   class nnUNetTrainer_customTrainerName(nnUNetTrainer_pickAnExistingTrainerToModify):
       def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                    device: torch.device = torch.device('cuda')):
           super().__init__(plans, configuration, fold, dataset_json, device)
           # Add your own modifications here
   
   # using a standardized function name so that SCT can import the class
   def get_trainer_class():
      return nnUNetTrainer_customTrainerName
   ```
- Choose an appropriate base class to modify, and an appropriate name for your trainer, and update the names above.
    - [`nnunetv2`'s guidelines](https://github.com/spinalcordtoolbox/nnUNet-neuropoly/blob/neuropoly-fork-patches/documentation/extending_nnunet.md):
        > If you intend to modify the training procedure (loss, sampling, data augmentation, lr scheduler, etc) then you need to implement your own trainer class. Best practice is to create a class that inherits from nnUNetTrainer and implements the necessary changes. Head over to our [trainer classes folder](https://github.com/spinalcordtoolbox/nnUNet-neuropoly/tree/neuropoly-fork-patches/nnunetv2/training/nnUNetTrainer) for inspiration! There will be similar trainers for what you intend to change and you can take them as a guide. nnUNetTrainer are structured similarly to PyTorch lightning trainers, this should also make things easier! 
    - Example: [`ms-lesion-agnostic/nnunet/trainer_class.py`](https://github.com/ivadomed/ms-lesion-agnostic/blob/main/nnunet/trainer_class.py).
- Copy the entire `trainer_class.py` file into the trainer classes folder.
    - If you have `git cloned` the `nnunetv2-neuropoly` repo, then this will be easy.
    - If you have installed via `pip` or `conda`, you will have to dig into the virtual environment to find the right folder. This will also depend on OS.
- You can now specify this trainer class in your [`plans.json`](https://github.com/spinalcordtoolbox/nnUNet-neuropoly/blob/neuropoly-fork-patches/documentation/explanation_plans_files.md) file, under the "`network_arch_class_name`" key.

> [!IMPORTANT]
> If you use a custom trainer, and you wish to package your model for use with SCT, you will need to include this `trainer_class.py` file when you distribute your model.
>
> So, we highly recommend that you also commit this `trainer_class.py` file to your repo so that it is easy to access and review.

### iii. Run training

``` 
CUDA_VISIBLE_DEVICES=X nnUNetv2_train DATASET_ID CONFIG FOLD
```

Replace `X` with GPU id for training.

Replace `DATASET_ID` with the same number as for `nnUNetv2_plan_and_preprocess` command.

Replace `CONFIG` with `2d`, `3d_fullres`, `3d_lowres`, or `3d_cascade_fullres` configuration.

Replace `FOLD` with 0 if you want to run only a single fold; otherwise, 5 folds are the default.

> **Note**
> Every 50 epochs, a checkpoint is saved (do not stop before the 50th epoch if you want to run inference). You can continue a previous training from the latest checkpoint, by adding the `--c` flag to the `nnUNetv2_train` command.

> **Note**
> Figure tracking the training progress is available `nnUNet_results/DATASET_ID/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_X/progress.png`
You can copy it locally using `scp PATH:server_file PATH:local_file`

----

## 5. Run prediction/inference

Only possible if 50+ epochs.

```
nnUNetv2_predict -i ${nnUNet_raw}/DATASET_ID/imagesTs -o OUT_DIR -d DATASET_ID -c CONFIG --save_probabilities -chk checkpoint_best.pth -f FOLD
```

Example of `OUT_DIR`: `${nnUNet_results}/<DATASET_NAME>/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/test`

----

## 6. Compute segmentation metrics

You can compute segmentation metrics (Dice, ...) using [our MetricsReloaded fork](https://github.com/ivadomed/MetricsReloaded/tree/main).

For details, see [MetricsReloaded quick start guide](https://github.com/ivadomed/MetricsReloaded/blob/main/MetricsReloaded_quick_start_guide.md).
