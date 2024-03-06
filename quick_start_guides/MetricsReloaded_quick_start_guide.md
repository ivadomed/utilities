# MetricsReloaded quick-start guide

Useful links:
- [MetricsReloaded GitHub page](https://github.com/Project-MONAI/MetricsReloaded)
- [MetricsReloaded documentation](https://metricsreloaded.readthedocs.io/en/latest/)
- [MetricsReloaded publication](https://www.nature.com/articles/s41592-023-02151-z)
- [MetricsReloaded preprint](https://arxiv.org/pdf/2206.01653v5.pdf) - preprint contains more figures than the publication

## Installation

Official installation instructions are available [here](https://github.com/Project-MONAI/MetricsReloaded?tab=readme-ov-file#installation).

> **Note**
> Always install MetricsReloaded inside a virtual environment.

```
# Create and activate a new conda environment
conda create -n metrics_reloaded python=3.10 pip
conda activate metrics_reloaded

# Clone the repository
cd ~/code
git clone https://github.com/csudre/MetricsReloaded.git
cd MetricsReloaded

# Install the package
python -m pip install .
# You can alternatively install the package in editable mode:
python -m pip install -e .
```

## Usage

You can use the [compute_metrics_reloaded.py](../compute_metrics/compute_metrics_reloaded.py) script to compute metrics using the MetricsReloaded package.

```commandline
python compute_metrics_reloaded.py -reference sub-001_T2w_seg.nii.gz -prediction sub-001_T2w_prediction.nii.gz 
```

Default metrics (semantic segmentation):
    - Dice similarity coefficient (DSC)
    - Normalized surface distance (NSD)
(for details, see Fig. 2, Fig. 11, and Fig. 12 in https://arxiv.org/abs/2206.01653v5)

The script is compatible with both binary and multi-class segmentation tasks (e.g., nnunet region-based).

The metrics are computed for each unique label (class) in the reference (ground truth) image.

The output is saved to a CSV file, for example:

```csv
reference   prediction	label	dsc	fbeta	nsd	vol_diff	rel_vol_diff	EmptyRef	EmptyPred
seg.nii.gz	pred.nii.gz	1.0	0.819	0.819	0.945	0.105	-10.548	False	False
seg.nii.gz	pred.nii.gz	2.0	0.743	0.743	0.923	0.121	-11.423	False	False
```