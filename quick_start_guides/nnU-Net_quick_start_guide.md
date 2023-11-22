# nnU-Net quick-start guide

This file provides a quick-start guide for nnU-Net v2.

nnU-Net is a self-configuring framework for deep learning-based medical image segmentation; see [nnUNet GitHub page](https://github.com/MIC-DKFZ/nnUNet) and [publication](https://www.nature.com/articles/s41592-020-01008-z).

## Installation

Official installation instructions are available [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md#installation-instructions).

> **Note**
> Always install nnU-Net inside a virtual environment.

> **Note**
> Run the installation commands on a GPU cluster.

You can use either `python -m venv` and `git clone`:

```console
cd ~
mkdir nnUNet_env
python -m venv nnUNet_env/
source nnUNet_env/bin/activate
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

Or `conda`:


```console
# create conda env
conda create --name nnunet
conda activate nnunet
```

GPU:

```console
# install pytorch using conda - https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# install nnunet
pip install nnunetv2
# Install hiddenlayer. hiddenlayer enables nnU-net to generate plots of the network topologies it generates
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

CPU (for inference only):

```console
# install pytorch - https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# install nnunet
pip install nnunetv2
# Install hiddenlayer. hiddenlayer enables nnU-net to generate plots of the network topologies it generates
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

## Environment variables

For details, see [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md#linux--macos).

nnU-Net requires the following three directories: `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`. You can create them using:

```console
cd <PATH_WHERE_YOU_WANT_TO_CREATE_THE_FOLDERS>
mkdir nnUNet_raw nnUNet_preprocessed nnUNet_results
```

Then, include variables with paths to these folders in your `~/.bashrc` or `~/.zshrc` file:

```
export nnUNet_raw="${HOME}/nnUNet_raw"
export nnUNet_preprocessed="${HOME}/nnUNet_preprocessed"
export nnUNet_results="${HOME}/nnUNet_results"
```

> **Note**
> Modify the paths according to where you created the folders.

## Data structure

nnU-Net expects the following data structure:

```
nnUNet_raw/Dataset001_NAME1
├── dataset.json
├── imagesTr
│   ├── BRATS_001_0000.nii.gz
│   ├── BRATS_001_0001.nii.gz
│   ├── ...
├── imagesTs
│   ├── BRATS_485_0000.nii.gz
│   ├── BRATS_485_0001.nii.gz
│   ├── ...
└── labelsTr
    ├── BRATS_001.nii.gz
    ├── BRATS_002.nii.gz
    ├── ...
```

- **imagesTr** contains the images belonging to the training cases. nnU-Net will perform pipeline configuration, training with 
cross-validation, as well as finding postprocessing and the best ensemble using this data. 
- **imagesTs** (optional) contains the images that belong to the test cases. nnU-Net does not use them! This could just 
be a convenient location for you to store these images. Remnant of the Medical Segmentation Decathlon folder structure.
- **labelsTr** contains the images with the ground truth segmentation maps for the training cases. 
- **dataset.json** contains metadata of the dataset.
  
You can use [our scripts](https://github.com/ivadomed/utilities/tree/main/dataset_conversion) to convert the data from BIDS to the nnU-Net format. 

## Train a model

> **Note**
> Always run training inside the virtual terminal. You can use [`screen`](https://intranet.neuro.polymtl.ca/geek-tips/bash-shell/README.html#screen-for-background-processes) or `tmux`.

1. Validate dataset integrity.
> Note that if you only plan to use 2d, 3d_fullres or 3d_lowres data, you should use the flag `-c <DATA_TYPE>` to only generate the wanted data and save some space ! (default: -c 2d 3d_fullres 3d_lowres)

> Also 3d_cascade_fullres uses 3d_fullres data

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -c 2d 3d_fullres 3d_lowres
```

Replace `DATASET_ID` with a number higher than 500, for example, `-d 501`.

2. Run training

``` 
CUDA_VISIBLE_DEVICES=X nnUNetv2_train DATASET_ID CONFIG FOLD
```

Replace `X` with GPU id for training.

Replace `DATASET_ID` with the same number as for `nnUNetv2_plan_and_preprocess` command.

Replace `CONFIG` with `2d`, `3d_fullres`, `3d_lowres`, or `3d_cascade_fullres` configuration.

Replace `FOLD` with 0 if you want to run only a single fold; otherwise, 5 folds are the default.

> **Note**
> Every 50 epochs, a checkpoint is saved (do not stop before the 50th epoch if you want to run inference).

> **Note**
> Figure tracking the training progress is available `nnUNet_results/DATASET_ID/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_X/progress.png`
You can copy it locally using `scp PATH:server_file PATH:local_file`


## Run prediction/inference

Only possible if 50+ epochs.

```
nnUNetv2_predict -i ${nnUNet_raw}/DATASET_ID/imagesTs -o OUT_DIR -d DATASET_ID -c CONFIG --save_probabilities -chk checkpoint_best.pth -f FOLD
```

Example of `OUT_DIR`: `${nnUNet_results}/<DATASET_NAME>/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/test`

## Compute ANIMA metrics

For MS and SCI lesion segmentation tasks, you can compute ANIMA metrics using the [compute_anima_metrics.py](https://github.com/ivadomed/model_seg_sci/blob/main/testing/compute_anima_metrics.py) script.

For details see [ANIMA quick start guide](ANIMA_quick_start_guide.md).
