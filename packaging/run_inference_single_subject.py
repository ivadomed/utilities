"""
This script is used to run inference on a single subject using a nnUNetV2 model.

Note: conda environment with nnUNetV2 is required to run this script.

1. Create a conda environment with the following command:

    conda create -n venv_nnunet python=3.9

2. Activate the environment with the following command:

    conda activate venv_nnunet

3. Install the required packages with the following command:

    cd <REPO>
    pip install -r packaging/requirements.txt


To temporarily suppress warnings raised by the nnUNet, you can run the following three commands in the same terminal
session as the above command:

    export nnUNet_raw="${HOME}/nnUNet_raw"
    export nnUNet_preprocessed="${HOME}/nnUNet_preprocessed"
    export nnUNet_results="${HOME}/nnUNet_results"

Note: the script contains reorientation of the input image to RPI orientation. This assumes that the model was trained
on images in RPI orientation. If the model was trained on images in a different orientation, the reorientation should
be modified/removed.

Authors: Jan Valosek, Naga Karthik

Example usage:
    python run_inference_single_subject.py
        -i sub-001_T2w.nii.gz
        -o sub-001_T2w_seg.nii.gz
        -path-model <PATH_TO_MODEL_FOLDER>
        -tile-step-size 0.5
"""


import os
import shutil
import subprocess
import argparse
import datetime

import torch
import glob
import time
import tempfile

from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data as predictor


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment an image using nnUNetV2 model.')
    parser.add_argument('-i', help='Input image to segment. Example: sub-001_T2w.nii.gz', required=True)
    parser.add_argument('-o', help='Output filename. Example: sub-001_T2w_seg.nii.gz', required=True)
    parser.add_argument('-path-model', help='Path to the model folder. This folder should contain individual '
                                            'folders like fold_0, fold_1, etc. and dataset.json, '
                                            'dataset_fingerprint.json and plans.json files.', required=True, type=str)
    parser.add_argument('-use-gpu', action='store_true', default=False,
                        help='Use GPU for inference. Default: False')
    parser.add_argument('-use-best-checkpoint', action='store_true', default=False,
                        help='Use the best checkpoint (instead of the final checkpoint) for prediction. '
                             'NOTE: nnUNet by default uses the final checkpoint. Default: False')
    parser.add_argument('-tile-step-size', default=0.5, type=float,
                        help='Tile step size defining the overlap between images patches during inference. '
                             'Default: 0.5 '
                             'NOTE: changing it from 0.5 to 0.9 makes inference faster but there is a small drop in '
                             'performance.')

    return parser


def get_orientation(file):
    """
    Get the original orientation of an image
    :param file: path to the image
    :return: orig_orientation: original orientation of the image, e.g. RPI
    """

    # Fetch the original orientation from the output of sct_image
    sct_command = "sct_image -i {} -header | grep -E qform_[xyz] | awk '{{printf \"%s\", substr($2, 1, 1)}}'".format(
        file)
    orig_orientation = subprocess.check_output(sct_command, shell=True).decode('utf-8')
    return orig_orientation


def tmp_create():
    """
    Create temporary folder and return its path
    """
    prefix = f"sciseg_prediction_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    print(f"Creating temporary folder ({tmpdir})")
    return tmpdir


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.
    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    Taken (shamelessly) from: https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py
    """
    dir, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            return os.path.join(dir, stem), ext
    # If no special case, behaves like the regular splitext
    stem, ext = os.path.splitext(filename)
    return os.path.join(dir, stem), ext


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension. Taken (shamelessly) from:
    https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py
    :param fname: absolute or relative file name. Example: t2.nii.gz
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii
    Examples:
    - add_suffix(t2.nii, _mean) -> t2_mean.nii
    - add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem + suffix + ext)


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Note: we use os.path.abspath to resolve relative paths
    fname_file = os.path.abspath(args.i)
    fname_file_out = os.path.abspath(args.o)
    print(f'\nFound {fname_file} file.')

    # Create temporary directory in the temp to store the reoriented images
    tmpdir = tmp_create()
    # Copy the file to the temporary directory using shutil.copyfile
    fname_file_tmp = os.path.join(tmpdir, os.path.basename(fname_file))
    shutil.copyfile(fname_file, fname_file_tmp)
    print(f'Copied {fname_file} to {fname_file_tmp}')

    # Get the original orientation of the image, for example RPI
    orig_orientation = get_orientation(fname_file_tmp)

    # Reorient the image to RPI orientation if not already in RPI
    if orig_orientation != 'RPI':
        print(f'Reorienting to RPI orientation...')
        # reorient the image to RPI using SCT
        os.system('sct_image -i {} -setorient RPI -o {}'.format(fname_file_tmp, fname_file_tmp))

    # NOTE: for individual images, the _0000 suffix is not needed.
    # BUT, the images should be in a list of lists
    fname_file_tmp_list = [[fname_file_tmp]]

    # Use all the folds available in the model folder by default
    folds_avail = [int(f.split('_')[-1]) for f in os.listdir(args.path_model) if f.startswith('fold_')]
    print(f'Using fold(s) {folds_avail}')

    # Create directory for nnUNet prediction
    tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
    fname_prediction = os.path.join(tmpdir_nnunet, os.path.basename(add_suffix(fname_file_tmp, '_pred')))
    os.mkdir(tmpdir_nnunet)

    # Run nnUNet prediction
    print('Starting inference...it may take a few minutes...')
    start = time.time()
    # directly call the predict function
    predictor(
        list_of_lists_or_source_folder=fname_file_tmp_list,
        output_folder=tmpdir_nnunet,
        model_training_output_dir=args.path_model,
        use_folds=folds_avail,
        tile_step_size=args.tile_step_size,     # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,      # applies gaussian noise and gaussian blur
        use_mirroring=False,    # test time augmentation by mirroring on all axes
        perform_everything_on_gpu=True if args.use_gpu else False,
        device=torch.device('cuda', 0) if args.use_gpu else torch.device('cpu'),
        verbose=False,
        save_probabilities=False,
        overwrite=True,
        checkpoint_name='checkpoint_final.pth' if not args.use_best_checkpoint else 'checkpoint_best.pth',
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3
    )
    end = time.time()

    print('Inference done.')
    total_time = end - start
    print('Total inference time: {} minute(s) {} seconds'.format(int(total_time // 60), int(round(total_time % 60))))

    # Check if the prediction file exists
    if not glob.glob(os.path.join(tmpdir_nnunet, '*.nii.gz')):
        raise FileNotFoundError(f'Prediction file not found in {tmpdir_nnunet}')
    # Copy .nii.gz file from tmpdir_nnunet to tmpdir
    pred_file = glob.glob(os.path.join(tmpdir_nnunet, '*.nii.gz'))[0]
    shutil.copyfile(pred_file, fname_prediction)

    print('Re-orienting the prediction back to original orientation...')
    # Reorient the image back to original orientation
    # skip if already in RPI
    if orig_orientation != 'RPI':
        print(f'Reorienting to original orientation {orig_orientation}...')
        # reorient the image to the original orientation using SCT
        os.system('sct_image -i {} -setorient {} -o {}'.format(fname_prediction, orig_orientation, fname_prediction))

    # Copy fname_prediction to fname_file_out
    shutil.copyfile(fname_prediction, fname_file_out)

    print('Deleting the temporary folder...')
    # Delete the temporary folder
    shutil.rmtree(tmpdir)

    print('-' * 50)
    print(f"Created {fname_file_out}")
    print('-' * 50)


if __name__ == '__main__':
    main()
