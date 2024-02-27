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

The output is saved to a JSON file, for example:

```json
{
    "reference": "sub-001_T2w_seg.nii.gz",
    "prediction": "sub-001_T2w_prediction.nii.gz",
    "1.0": {
        "dsc": 0.8195991091314031,
        "nsd": 0.9455782312925171
    },
    "2.0": {
        "dsc": 0.8042553191489362,
        "nsd": 0.9580573951434879
    }

}
```