# nnU-Net quick-start guide

This file provides a quick-start guide for nnU-Net v2.

nnU-Net is a self-configuring framework for deep learning-based medical image segmentation; see [nnUNet GitHub page](https://github.com/MIC-DKFZ/nnUNet) and [publication](https://www.nature.com/articles/s41592-020-01008-z).

## Installation

Official installation instructions are available [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md#installation-instructions).

> **Note**
> Always install nnU-Net inside a virtual environment.

> **Note**
> Run the installation commands on a GPU cluster, not on your laptop.

---

### `git clone`

`python -m venv` and `git clone`:

```console
cd ~
mkdir nnUNet_env
python -m venv nnUNet_env/
source nnUNet_env/bin/activate
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

---

### `conda`

```console
# create conda env
conda create --name nnunet python=3.9
conda activate nnunet
```

**GPU `conda install`:**

```console
# install pytorch using conda - https://pytorch.org/get-started/locally/
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
# install nnunet
pip install nnunetv2
# Install hiddenlayer. hiddenlayer enables nnU-net to generate plots of the network topologies it generates
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

**GPU `pip3 install`:**

```console
# install pytorch using pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2
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

```console
# install pytorch - https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# install nnunet
pip install nnunetv2
# Install hiddenlayer. hiddenlayer enables nnU-net to generate plots of the network topologies it generates
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

---

### upgrades

To upgrade nnunetv2 to the latest version, you can run the following command in your virtual env:

```console
pip install --upgrade nnunetv2
```

To check the current version and to upgrade to a specific version, you can use: 

```console
pip freeze | grep nnunet
pip install nnunetv2==2.4.1
```

## Environment variables

For details, see [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md#linux--macos).

nnU-Net requires the following three directories: `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`. You can create them using the commands below.

> **Note**
> Typically, these folders need to be created on the GPU server, not on the computer. You can connect to our GPU servers using `ssh`; see the intranet for details.

```console
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

## Data structure

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

## Train a model

> **Note**
> Since you will likely be running the training on one of our GPU servers, you will need to get your training data there. See [our intranet](https://intranet.neuro.polymtl.ca/computing-resources/neuropoly/gpus.html#data) for details. 

> **Note**
> Always run training inside the virtual terminal. You can use [`screen`](https://intranet.neuro.polymtl.ca/geek-tips/bash-shell/README.html#screen-for-background-processes) or `tmux`.

1. Validate dataset integrity.
> Note that if you only plan to use 2d, 3d_fullres or 3d_lowres data, you should use the flag `-c <DATA_TYPE>` to only generate the wanted data and save some space! (default: -c 2d 3d_fullres 3d_lowres)

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
> Every 50 epochs, a checkpoint is saved (do not stop before the 50th epoch if you want to run inference). You can continue a previous training from the latest checkpoint, by adding the `--c` flag to the `nnUNetv2_train` command.

> **Note**
> Figure tracking the training progress is available `nnUNet_results/DATASET_ID/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_X/progress.png`
You can copy it locally using `scp PATH:server_file PATH:local_file`


## Run prediction/inference

Only possible if 50+ epochs.

```
nnUNetv2_predict -i ${nnUNet_raw}/DATASET_ID/imagesTs -o OUT_DIR -d DATASET_ID -c CONFIG --save_probabilities -chk checkpoint_best.pth -f FOLD
```

Example of `OUT_DIR`: `${nnUNet_results}/<DATASET_NAME>/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/test`

## Compute segmentation metrics

You can compute segmentation metrics (Dice, ...) using [our MetricsReloaded fork](https://github.com/ivadomed/MetricsReloaded/tree/main).

For details, see [MetricsReloaded quick start guide](https://github.com/ivadomed/MetricsReloaded/blob/main/MetricsReloaded_quick_start_guide.md).
