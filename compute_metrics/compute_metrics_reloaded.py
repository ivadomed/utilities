"""
Compute MetricsReloaded metrics for segmentation tasks.
Details: https://github.com/Project-MONAI/MetricsReloaded/tree/main

Example usage:
    python compute_metrics_reloaded.py
        -reference sub-001_T2w_seg.nii.gz
        -prediction sub-001_T2w_prediction.nii.gz

Default metrics (semantic segmentation):
    - Dice similarity coefficient (DSC)
    - Normalized surface distance (NSD)
(for details, see Fig. 2, Fig. 11, and Fig. 12 in https://arxiv.org/abs/2206.01653v5)

The script is compatible with both binary and multi-class segmentation tasks (e.g., nnunet region-based).
The metrics are computed for each unique label (class) in the reference (ground truth) image.
The output is saved to a JSON file, for example:

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

Authors: Jan Valosek
"""


import os
import argparse
import json
import numpy as np
import nibabel as nib

from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as BPM


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Compute MetricsReloaded metrics for segmentation tasks.')

    # Arguments for model, data, and training
    parser.add_argument('-prediction', required=True, type=str,
                        help='Path to the nifti image of test prediction.')
    parser.add_argument('-reference', required=True, type=str,
                        help='Path to the nifti image of reference (ground truth) label.')
    parser.add_argument('-metrics', nargs='+', default=['dsc', 'nsd'], required=False,
                        help='List of metrics to compute. For details, '
                             'see: https://metricsreloaded.readthedocs.io/en/latest/reference/metrics/metrics.html. '
                             'Default: dsc, nsd')
    parser.add_argument('-output', type=str, default='metrics.json', required=False,
                        help='Path to the output JSON file to save the metrics. Default: metrics.json')

    return parser


def load_nifti_image(file_path):
    """
    Construct absolute path to the nifti image, check if it exists, and load the image data.
    :param file_path: path to the nifti image
    :return: nifti image data
    """
    file_path = os.path.expanduser(file_path)   # resolve '~' in the path
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} does not exist.')
    nifti_image = nib.load(file_path)
    return nifti_image.get_fdata()


def main():

    # parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # load nifti images
    prediction_data = load_nifti_image(args.prediction)
    reference_data = load_nifti_image(args.reference)

    # check whether the images have the same shape and orientation
    if prediction_data.shape != reference_data.shape:
        raise ValueError(f'The prediction and reference (ground truth) images must have the same shape. '
                         f'The prediction image has shape {prediction_data.shape} and the ground truth image has '
                         f'shape {reference_data.shape}.')

    # get all unique labels (classes)
    # for example, for nnunet region-based segmentation, spinal cord has label 1, and lesions have label 2
    unique_labels_reference = np.unique(reference_data)
    unique_labels_reference = unique_labels_reference[unique_labels_reference != 0]  # remove background label

    # create dictionary to store the metrics
    output_dict = {'reference': args.reference, 'prediction': args.prediction}

    # loop over all unique labels
    for label in unique_labels_reference:
        # create binary masks for the current label
        print(f'Processing label {label}')
        predidction_data_label = np.array(prediction_data == label, dtype=float)
        reference_data_label = np.array(reference_data == label, dtype=float)

        # Dice similarity coefficient (DSC):
        # Fig. 65 in https://arxiv.org/pdf/2206.01653v5.pdf
        # https://metricsreloaded.readthedocs.io/en/latest/reference/metrics/pairwise_measures.html#MetricsReloaded.metrics.pairwise_measures.BinaryPairwiseMeasures.dsc
        # Normalized surface distance (NSD):
        # Fig. 86 in https://arxiv.org/pdf/2206.01653v5.pdf
        # https://metricsreloaded.readthedocs.io/en/latest/reference/metrics/pairwise_measures.html#MetricsReloaded.metrics.pairwise_measures.BinaryPairwiseMeasures.normalised_surface_distance
        bpm = BPM(predidction_data_label, reference_data_label, measures=args.metrics)
        dict_seg = bpm.to_dict_meas()

        # add the metrics to the output dictionary
        output_dict[label] = dict_seg

    # save dict as json
    fname_output = os.path.abspath(args.output)
    with open(fname_output, 'w') as f:
        json.dump(output_dict, f, indent=4)
    print(f'Saved metrics to {fname_output}.')


if __name__ == '__main__':
    main()
