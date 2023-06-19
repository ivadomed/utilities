"""
Converts nnUNetv2 dataset format to the BIDS-structured dataset. Full details about
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Note that the conversion from BIDS to nnUNet is done using symbolic links to avoid creating multiple copies of the
(original) BIDS dataset.

Currently only supports the conversion of a single contrast. In case of multiple contrasts, the script should be
modified to include those as well.

Usage example:
    python convert_bids_to_nnUNetv2.py --path-data ~/data/dataset --path-out ~/data/dataset-nnunet
                    --dataset-name MyDataset --dataset-number 501 --split 0.6 0.2 --seed 99

Naga Karthik, Jan Valosek modified by Théo Mathieu
"""