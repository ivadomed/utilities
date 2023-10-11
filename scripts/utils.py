import os
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np

from utils import fetch_contrast
from image import Image

## Global variables
CONTRAST = {'t1': ['T1w'],
            't2': ['T2w'],
            't2s':['T2star'],
            't1_t2': ['T1w', 'T2w']}

## Functions
def get_img_path_from_mask_path(str_path):
    """
    This function does 2 things: ⚠️ Files need to be stored in a BIDS compliant dataset
        - Step 1: Remove label suffix (e.g. "_labels-disc-manual"). The suffix is always between the MRI contrast and the file extension.
        - Step 2: Remove derivatives path (e.g. derivatives/labels/). The first folders is always called derivatives but the second may vary (e.g. labels_soft)

    :param path: absolute path to the label img. Example: /<path_to_BIDS_data>/derivatives/labels/sub-amuALT/anat/sub-amuALT_T1w_labels-disc-manual.nii.gz
    :return: img path. Example: /<path_to_BIDS_data>/sub-amuALT/anat/sub-amuALT_T1w.nii.gz
    Based on https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    """
    # Load path
    path = Path(str_path)

    # Extract file extension
    ext = ''.join(path.suffixes)

    # Get img name
    img_name = '_'.join(path.name.split('_')[:-1]) + ext
    
    # Create a list of the directories
    dir_list = str(path.parent).split('/')

    # Remove "derivatives" and "labels" folders
    derivatives_idx = dir_list.index('derivatives')
    dir_path = '/'.join(dir_list[0:derivatives_idx] + dir_list[derivatives_idx+2:])

    # Recreate img path
    img_path = os.path.join(dir_path, img_name)

    return img_path
    
##
def get_mask_path_from_img_path(img_path, suffix='_seg', derivatives_path='/derivatives/labels'):
    """
    This function returns the mask path from an image path. Images need to be stored in a BIDS compliant dataset.

    :param img_path: String path to niftii image
    :param suffix: Mask suffix
    :param derivatives_path: Relative path to derivatives folder where labels are stored (e.i. '/derivatives/labels')
    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    """
    # Extract information from path
    subjectID, sessionID, filename, contrast, echoID = fetch_subject_and_session(img_path)

    # Extract file extension
    path_obj = Path(img_path)
    ext = ''.join(path_obj.suffixes)

    # Create mask name
    mask_name = path_obj.name.split('.')[0] + suffix + ext

    # Split path using "/" (TODO: check if it works for windows users)
    path_list = img_path.split('/')

    # Extract subject folder index
    sub_folder_idx = path_list.index(subjectID)

    # Reconstruct mask_path
    mask_path = os.path.join('/'.join(path_list[:sub_folder_idx]), derivatives_path, path_list[sub_folder_idx:-1], mask_name)
    return mask_path

##
def change_mask_suffix(mask_path, new_suffix='_seg'):
    """
    This function replace the current suffix with a new suffix suffix. If path is specified, make sure the dataset is BIDS compliant.

    :param mask_path: Input mask filepath or filename
    :param new_suffix: New mask suffix
    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    """

    # Extract file extension
    ext = ''.join(Path(mask_path).suffixes)

    # Change mask path
    new_mask_path = '_'.join(mask_path.split('_')[:-1]) + new_suffix + ext
    return new_mask_path

##
def fetch_subject_and_session(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subjectID: subject ID (e.g., sub-001)
    :return: sessionID: session ID (e.g., ses-01)
    :return: filename: nii filename (e.g., sub-001_ses-01_T1w.nii.gz)
    :return: contrast: MRI modality (dwi or anat)
    :return: echoID: echo ID (e.g., echo-1)
    :return: acquisition: acquisition (e.g., acq_sag)
    Based on https://github.com/spinalcordtoolbox/manual-correction
    """

    _, filename = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    subject = re.search('sub-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash

    session = re.search('ses-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    sessionID = session.group(0)[:-1] if session else ""    # [:-1] removes the last underscore or slash

    echo = re.search('echo-(.*?)[_]', filename_path)     # [_/] means either underscore or slash
    echoID = echo.group(0)[:-1] if echo else ""    # [:-1] removes the last underscore or slash

    acq = re.search('acq-(.*?)[_]', filename_path)     # [_/] means either underscore or slash
    acquisition = acq.group(0)[:-1] if acq else ""    # [:-1] removes the last underscore or slash
    # REGEX explanation
    # . - match any character (except newline)
    # *? - match the previous element as few times as possible (zero or more times)

    contrast = 'dwi' if 'dwi' in filename_path else 'anat'  # Return contrast (dwi or anat)

    return subjectID, sessionID, filename, contrast, echoID, acquisition


def fetch_contrast(filename_path):
    '''
    Extract MRI contrast from a BIDS-compatible filename/filepath
    The function handles images only.
    :param filename_path: image file path or file name. (e.g sub-001_ses-01_T1w.nii.gz)
    Copied from https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    '''
    return filename_path.rstrip(''.join(Path(filename_path).suffixes)).split('_')[-1]


def edit_metric_dict(metrics_dict, img_path, mask_path, disc_label_suffix='_labels-disc-manual'):
    '''
    This function extracts information and metadata from an image and its mask. Values are then
    gathered inside a dictionary.

    :param metrics_dict: dictionary where information will be gathered
    :param img_path: niftii image path
    :param discs_mask_path: corresponding niftii discs mask path

    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    '''
    #-----------------------------------------------------------------------#
    #----------------------- Extracting metadata ---------------------------#
    #-----------------------------------------------------------------------#

    # Extract field of view information thanks to discs labels
    if '_labels-disc' in mask_path:
        discs_mask_path = mask_path
    else:
        discs_mask_path = change_mask_suffix(mask_path, new_suffix=disc_label_suffix)

    if os.path.exists(discs_mask_path):
        discs_mask = Image(discs_mask_path)
        disc_list = [list(coord)[-1] for coord in discs_mask.getNonZeroCoordinates(sorting='value')]
    else:
        disc_list = []
    
    # Extract original image orientation 
    img = Image(img_path)
    orientation = img.get_orientation

    # Extract image dimensions and resolutions
    img_RSP = img.change_orientation("RSP")
    nx, ny, nz, nt, px, py, pz, pt = img_RSP.get_dimension

    # Check for shape mismatch between mask and image
    if img.data.shape != Image(mask_path).data.shape:
        shape_mismatch = True
    else:
        shape_mismatch = False

    # Extract MRI contrast
    contrast = fetch_contrast(img_path)

    #-----------------------------------------------------------------------#
    #--------------------- Adding metadata to dict -------------------------#
    #-----------------------------------------------------------------------#
    list_of_metrics = [img_path, orientation, contrast, disc_list, shape_mismatch, nx, ny, nz, nt, px, py, pz, pt]
    list_of_keys = ['img_path', 'orientation', 'contrast', 'disc_list', 'shape_mismatch', 'nx', 'ny', 'nz', 'nt', 'px', 'py', 'pz', 'pt']
    for key, metric in zip(list_of_keys, list_of_metrics):
        if type(metric) != list():
            metric = [metric]
        if key not in metrics_dict.keys():
            metrics_dict[key] = metric
        else:
            metrics_dict[key] += metric
    
    return metrics_dict


def save_violin(splits, values, output_path, y_axis):
    '''
    Create a violin plot
    :param splits: String list of the split name
    :param values: Values associated with the split
    :param output_path: Path to output folder where figures will be stored
    :param y_axis: y-axis name

    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    '''
            
    # Set position of bar on X axis
    result_dict = {}
    for i, split in enumerate(splits):
        result_dict[split]=values[i]
    result_df = pd.DataFrame(data=result_dict)

    # Make the plot 
    plot = sns.violinplot(data=result_df)  
    plot.set(xlabel='split', ylabel=y_axis)
    plot.set(title=y_axis)
    
    # Save plot
    plot.figure.savefig(output_path)


def save_graphs(output_folder, metrics_dict):
    '''
    Plot and save metrics into an output folder

    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    '''
    # Extract subjects and metrics
    splits = np.array(list(metrics_dict.keys()))
    metrics_names = list(metrics_dict[splits[0]].keys())

    # Use violin plots
    for metric in ['disc_list', 'nx', 'ny', 'nz', 'nt', 'px', 'py', 'pz', 'pt']:
        out_path = os.path.join(output_folder, f'{metric}.png')
        save_violin(splits=splits, values=[metrics_dict[split][metric] for split in splits], output_path=out_path, y_axis=metric)

    # Use bar graphs


