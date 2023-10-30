import os
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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


def edit_metric_dict(metrics_dict, img_path, mask_path, discs_mask_path):
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

    if os.path.exists(discs_mask_path): # TODO: deal with datasets with no discs labels
        discs_mask = Image(discs_mask_path)
        discs_labels = [list(coord)[-1] for coord in discs_mask.getNonZeroCoordinates(sorting='value')]
    else:
        discs_labels = []
    
    # Extract original image orientation 
    img = Image(img_path)
    orientation = img.orientation

    # Extract image dimensions and resolutions
    img_RPI = img.change_orientation("RPI")
    nx, ny, nz, nt, px, py, pz, pt = img_RPI.dim

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
    list_of_metrics = [img_path, orientation, contrast, discs_labels, shape_mismatch, nx, ny, nz, nt, px, py, pz, pt]
    list_of_keys = ['img_path', 'orientation', 'contrast', 'discs_labels', 'shape_mismatch', 'nx', 'ny', 'nz', 'nt', 'px', 'py', 'pz', 'pt']
    for key, metric in zip(list_of_keys, list_of_metrics):
        if not isinstance(metric, list):
            metric = [metric]
        if key not in metrics_dict.keys():
            metrics_dict[key] = metric
        else:
            metrics_dict[key] += metric
    
    return metrics_dict


def save_violin(names, values, output_path, x_axis, y_axis):
    '''
    Create a violin plot
    :param names: String list of the names
    :param values: Values associated with the names
    :param output_path: Output path (string)
    :param x_axis: x-axis name
    :param y_axis: y-axis name

    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    '''
            
    # Set position of bar on X axis
    result_dict = {'names':[], 'values':[]}
    for i, name in enumerate(names):
        result_dict['values'] += values[i]
        for j in range(len(values[i])):
            result_dict['names'] += [name] 
    
    result_df = pd.DataFrame(data=result_dict)

    # Make the plot
    plt.figure()
    sns.violinplot(x="names", y="values", data=result_df)
    plt.xlabel(x_axis, fontsize = 15)
    plt.ylabel(y_axis, fontsize = 15)
    plt.title(y_axis, fontsize = 20)
    
    # Save plot
    plt.savefig(output_path)


def save_hist(names, values, output_path, x_axis, y_axis):
    '''
    Create a histogram plot
    :param names: String list of the names
    :param values: Values associated with the names
    :param output_path: Output path (string)
    :param x_axis: x-axis name
    :param y_axis: y-axis name

    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    '''
            
    # Set position of bar on X axis
    result_dict = {'names':[], 'values':[]}
    for i, name in enumerate(names):
        result_dict['values'] += values[i]
        for j in range(len(values[i])):
            result_dict['names'] += [name] 
    
    result_df = pd.DataFrame(data=result_dict)

    # Make the plot
    plt.figure()
    sns.histplot(data=result_df, x="values", hue="names", multiple="dodge", binwidth=1/len(names))
    plt.xlabel(x_axis, fontsize = 15)
    plt.xticks(np.arange(1, np.max(result_dict['values'])+1))
    plt.ylabel(x_axis, fontsize = 15)
    plt.title(y_axis, fontsize = 20)
    
    # Save plot
    plt.savefig(output_path)


def save_pie(names, values, output_path, x_axis, y_axis):
    '''
    Create a pie chart plot
    :param names: String list of the names
    :param values: Values associated with the names
    :param output_path: Output path (string)
    :param x_axis: x-axis name
    :param y_axis: y-axis name

    Based on https://www.geeksforgeeks.org/how-to-create-a-pie-chart-in-seaborn/ 
    '''
    # Set position of bar on X axis
    result_dict = {}
    for i, name in enumerate(names):
        result_dict[name] = {}
        for val in values[i]:
            if val not in result_dict[name].keys():
                result_dict[name][val] = 1
            else:
                result_dict[name][val] += 1
    
    # define Seaborn color palette to use 
    palette_color = sns.color_palette('bright')

    def autopct_format(values):
        '''
        Based on https://stackoverflow.com/questions/53782591/how-to-display-actual-values-instead-of-percentages-on-my-pie-chart-using-matplo
        '''
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total)/100)
            return '{v:d}'.format(v=val)
        return my_format

    # Make the plot
    if len(names) == 1:
        fig = plt.figure()
        plt.pie(result_dict[names[0]].values(), labels=result_dict[names[0]].keys(), colors=palette_color, autopct=autopct_format(result_dict[names[0]].values()))
        plt.title(y_axis, fontsize = 20)
        plt.xlabel(x_axis, fontsize = 15)
        plt.ylabel(y_axis, fontsize = 15)
    else:
        fig, axs = plt.subplots(1, len(names), figsize=(3*len(names),5))
        fig.suptitle(y_axis)

        for j, name in enumerate(result_dict.keys()):
            axs[j].pie(result_dict[name].values(), labels=result_dict[name].keys(), colors=palette_color, autopct=autopct_format(result_dict[names[0]].values()))
            axs[j].set_title(name)
        
        for ax, name in zip(axs.flat, names):
            ax.set(xlabel=name, ylabel=y_axis)
        
    # Save plot
    plt.savefig(output_path)


def save_graphs(output_folder, metrics_dict, data_type='split'):
    '''
    Plot and save metrics into an output folder

    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    '''
    # Extract subjects and metrics
    data_name = np.array(list(metrics_dict.keys()))
    metrics_names = list(metrics_dict[data_name[0]].keys())

    # Use violin plots
    for metric in ['nx', 'ny', 'nz', 'nt', 'px', 'py', 'pz', 'pt']:
        out_path = os.path.join(output_folder, f'{metric}.png')
        save_violin(names=data_name, values=[metrics_dict[name][metric] for name in data_name], output_path=out_path, x_axis=data_type, y_axis=metric)

    # Use bar pie chart
    for metric in ['orientation', 'contrast']:
        out_path = os.path.join(output_folder, f'{metric}.png')
        save_pie(names=data_name, values=[metrics_dict[name][metric] for name in data_name], output_path=out_path, x_axis=data_type, y_axis=metric)

    # Use bar graphs
    for metric in ['discs_labels']:
        out_path = os.path.join(output_folder, f'{metric}.png')
        save_hist(names=data_name, values=[metrics_dict[name][metric] for name in data_name], output_path=out_path, x_axis=metric, y_axis='Count')
