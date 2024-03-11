import os
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import math

from image import Image

## Global variables
CONTRAST = {'t1': ['T1w'],
            't2': ['T2w'],
            't2s':['T2star'],
            't1_t2': ['T1w', 'T2w'],
            'psir': ['PSIR'],
            'stir': ['STIR'],
            'psir_stir': ['PSIR', 'STIR'],
            't1_t2_psir_stir': ['T1w', 'T2w', 'PSIR', 'STIR']
            }

## Functions
def get_img_path_from_mask_path(str_path, derivatives_folder='derivatives'):
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
    derivatives_idx = dir_list.index(derivatives_folder)
    dir_path = '/'.join(dir_list[0:derivatives_idx] + dir_list[derivatives_idx+2:])

    # Recreate img path
    img_path = os.path.join(dir_path, img_name)

    return img_path
    
##
def get_mask_path_from_img_path(img_path, deriv_sub_folders, short_suffix='_seg', ext='.nii.gz', counterexample=[]):
    """
    This function returns the mask path from an image path or an empty string if the path does not exists. Images need to be stored in a BIDS compliant dataset.

    :param img_path: String path to niftii image
    :param suffix: Mask suffix
    :param ext: File extension

    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    """
    # Extract information from path
    subjectID, sessionID, filename, contrast, echoID, acq = fetch_subject_and_session(img_path)

    # Find corresponding mask
    mask_path = []
    for deriv_path in deriv_sub_folders:
        if counterexample: # Deal with counter examples
            paths = []
            for path in glob.glob(deriv_path + filename.split(ext)[0] + short_suffix + "*" + ext):
                iswrong = False
                for c in counterexample:
                    if c in path:
                        iswrong = True
                if not iswrong:
                    paths.append(path)
        else:
            paths = glob.glob(deriv_path + filename.split(ext)[0] + "*" + short_suffix + "*" + ext)

        if len(paths) > 1:
            print(f'Image {img_path} has multiple masks\n: {'\n'.join(paths)}')
        elif len(paths) == 1:
            mask_path.append(paths[0])
    return mask_path


def get_cont_path_from_other_cont(str_path, cont):
    """
    :param str_path: absolute path to the input nifti img. Example: /<path_to_BIDS_data>/sub-amuALT/anat/sub-amuALT_T1w.nii.gz
    :param cont: contrast of the target output image stored in the same data folder. Example: T2w
    :return: path to the output target image. Example: /<path_to_BIDS_data>/sub-amuALT/anat/sub-amuALT_T2w.nii.gz

    """
    # Load path
    path = Path(str_path)

    # Extract file extension
    ext = ''.join(path.suffixes)

    # Remove input contrast from name
    path_list = path.name.split('_')
    suffixes_pos = [1 if len(part.split('-')) == 1 else 0 for part in path_list]
    contrast_idx = suffixes_pos.index(1) # Find suffix

    # New image name
    img_name = '_'.join(path_list[:contrast_idx]+[cont]) + ext

    # Recreate img path
    img_path = os.path.join(str(path.parent), img_name)

    return img_path

def get_deriv_sub_from_img_path(img_path, derivatives_folder='derivatives'):
    """
    This function returns the derivatives path of the subject from an image path or an empty string if the path does not exists. Images need to be stored in a BIDS compliant dataset.

    :param img_path: String path to niftii image
    :param derivatives_folder: List of derivatives paths
    :param ext: File extension
    """
    # Extract information from path
    subjectID, sessionID, filename, contrast, echoID, acq = fetch_subject_and_session(img_path)
    path_bids, path_sub_folder = img_path.split(subjectID)[0:-1]
    path_sub_folder = subjectID + path_sub_folder

    # Find corresponding mask
    deriv_sub_folder = glob.glob(path_bids + "**/" + derivatives_folder + "/**/" + path_sub_folder, recursive=True)

    return deriv_sub_folder

##
def change_mask_suffix(mask_path, short_suffix='_seg', ext='.nii.gz'):
    """
    This function replace the current suffix with a new suffix suffix. If path is specified, make sure the dataset is BIDS compliant.

    :param mask_path: Input mask filepath or filename
    :param new_suffix: New mask suffix
    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    """
    # Extract information from path
    subjectID, sessionID, filename, contrast, echoID, acq = fetch_subject_and_session(mask_path)
    path_deriv_sub = mask_path.split(filename)[0]

    # Find corresponding new_mask
    new_mask_path = glob.glob(path_deriv_sub + '_'.join(filename.split('_')[:-1]) + short_suffix + "*" + ext)

    if len(new_mask_path) > 1:
        print(f'Multiple {short_suffix} masks for subject {subjectID} \n: {'\n'.join(new_mask_path)}')
        mask_path = ''
    elif len(new_mask_path) == 1:
        new_mask_path = new_mask_path[0]
    else: # mask does not exist
        new_mask_path = ''

    return new_mask_path


def list_der_suffixes(folder_path, ext='.nii.gz'):
    """
    This function return all the labels suffixes. If path is specified, make sure the dataset is BIDS compliant.

    :param folder_path: Path to folder where labels are stored.
    """
    folder_path = os.path.normpath(folder_path) 
    files = [file for file in os.listdir(folder_path) if file.endswith(ext)]
    suffixes = []
    for file in files:
        subjectID, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(file)
        split_file = file.split(ext)[0].split('_')
        skip_idx = 0
        for sp in [subjectID, sessionID, echoID, acquisition]:
            if sp:
                skip_idx = skip_idx + 1
        suffix = '_' + '_'.join(split_file[skip_idx+1:]) # +1 to skip contrast
        if not suffix =='_':
            suffixes.append(suffix)
    return suffixes
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
    Extract MRI contrast from a BIDS-compatible IMAGE filename/filepath
    The function handles images only.
    :param filename_path: image file path or file name. (e.g sub-001_ses-01_T1w.nii.gz)
    Copied from https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    '''
    return filename_path.rstrip(''.join(Path(filename_path).suffixes)).split('_')[-1]

def str_to_str_list(string):
    string = string[1:-1] # remove brackets
    return [s[1:-1] for s in string.split(', ')]

def str_to_float_list(string):
    string = string[1:-1] # remove brackets
    return [float(s) for s in string.split(', ')]


def edit_metric_dict(metrics_dict, fprint_dict, img_path, seg_paths, discs_paths, deriv_sub_folders):
    '''
    This function extracts information and metadata from an image and its mask. Values are then
    gathered inside a dictionary.

    :param metrics_dict: dictionary containing summary metadata
    :param fprint_dict: dictionary containing all the informations
    :param img_path: niftii image path
    :param seg_path: corresponding niftii spinal cord segmentation path
    :param discs_path: corresponding niftii discs mask path

    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    '''
    #-----------------------------------------------------------------------#
    #----------------------- Extracting metadata ---------------------------#
    #-----------------------------------------------------------------------#
    # Extract original image orientation 
    img = Image(img_path)
    orientation = img.orientation

    # Extract information from path
    subjectID, sessionID, filename, c, echoID, acq = fetch_subject_and_session(img_path)

    # Extract image dimensions and resolutions
    img_RPI = img.change_orientation("RPI")
    nx, ny, nz, nt, px, py, pz, pt = img_RPI.dim
    
    # Extract discs + check for shape mismatch between discs labels and image
    discs_labels = []
    count_discs = 0
    if discs_paths:
        for path in discs_paths:
            discs_mask = Image(path).change_orientation("RPI")
            discs_labels += [list(coord)[-1] for coord in discs_mask.getNonZeroCoordinates(sorting='value')]
            if img_RPI.data.shape != discs_mask.data.shape:
                count_discs += 1

    # Check for shape mismatch between segmentation and image
    count_seg = 0
    if seg_paths:
        for path in seg_paths:
            if img_RPI.data.shape != Image(path).change_orientation("RPI").data.shape:
                count_seg += 1

    # Compute image size
    X, Y, Z = nx*px, ny*py, nz*pz

    # Extract MRI contrast from image only
    contrast = fetch_contrast(img_path)

    # Extract suffixes
    suffixes = []
    for path in deriv_sub_folders:
        for suf in list_der_suffixes(path):
            if not suf in suffixes:
                suffixes.append(suf)
    
    # Extract derivatives folder
    der_folders = []
    for path in deriv_sub_folders:
        der_folders.append(os.path.basename(os.path.dirname(path.split(subjectID)[0])))

    #-------------------------------------------------------------------------------#
    #--------------------- Adding metadata to summary dict -------------------------#
    #-------------------------------------------------------------------------------#
    list_of_metrics = [orientation, contrast, X, Y, Z, nx, ny, nz, nt, px, py, pz, pt]
    list_of_keys = ['orientation', 'contrast', 'X', 'Y', 'Z', 'nx', 'ny', 'nz', 'nt', 'px', 'py', 'pz', 'pt']
    for key, metric in zip(list_of_keys, list_of_metrics):
        if not isinstance(metric,str):
            metric = str(metric)
        if key not in metrics_dict.keys():
            metrics_dict[key] = {metric:1}
        else:
            if metric not in metrics_dict[key].keys():
                metrics_dict[key][metric] = 1
            else:
                metrics_dict[key][metric] += 1

    # Add count shape mismatch
    key_mis_seg = 'mismatch-seg'
    if key_mis_seg not in metrics_dict.keys():
        metrics_dict[key_mis_seg] = count_seg
    else:
        metrics_dict[key_mis_seg] += count_seg

    key_mis_disc = 'mismatch-disc'
    if key_mis_disc not in metrics_dict.keys():
        metrics_dict[key_mis_disc] = count_discs
    else:
        metrics_dict[key_mis_disc] += count_discs

    # Add discs labels
    key_discs = 'discs-labels'
    if discs_labels:
        if key_discs not in metrics_dict.keys():
            metrics_dict[key_discs] = {}
        for disc in discs_labels:
            disc = str(disc)
            if disc not in metrics_dict[key_discs].keys():
                metrics_dict[key_discs][disc] = 1
            else:
                metrics_dict[key_discs][disc] += 1

    # Add suffixes
    suf_key = 'suffixes'
    if suf_key not in metrics_dict.keys():
        metrics_dict[suf_key] = suffixes
    else:
        for suf in suffixes:
            if not suf in metrics_dict[suf_key]:
                metrics_dict[suf_key].append(suf)
    
    # Add derivatives folders
    der_key = 'derivatives'
    if der_key not in metrics_dict.keys():
        metrics_dict[der_key] = der_folders
    else:
        for der in der_folders:
            if not der in metrics_dict[der_key]:
                metrics_dict[der_key].append(der)
    
    #--------------------------------------------------------------------------------#
    #--------------------- Storing metadata to exhaustive dict -------------------------#
    #--------------------------------------------------------------------------------#
    fprint_dict[filename] = {}

    # Add contrast
    fprint_dict[filename]['contrast'] = contrast
    
    # Add orientation
    fprint_dict[filename]['img_orientation'] = orientation

    # Add info SC segmentations
    if seg_paths:
        fprint_dict[filename]['seg-sc'] = True
        suf_seg = [path.split(contrast)[-1].split('.')[0] for path in seg_paths]
        fprint_dict[filename]['seg-suffix'] = '/'.join(suf_seg)
        fprint_dict[filename]['seg-mismatch'] = count_seg
    else:
        fprint_dict[filename]['seg-sc'] = False
        fprint_dict[filename]['seg-suffix'] = ''
        fprint_dict[filename]['seg-mismatch'] = count_seg
    
    # Add info discs labels
    if discs_paths:
        fprint_dict[filename]['discs-label'] = True
        suf_discs = [path.split(contrast)[-1].split('.')[0] for path in discs_paths]
        fprint_dict[filename]['discs-suffix'] = '/'.join(suf_discs)
        fprint_dict[filename]['discs-mismatch'] = count_discs
    else:
        fprint_dict[filename]['discs-label'] = False
        fprint_dict[filename]['discs-suffix'] = ''
        fprint_dict[filename]['discs-mismatch'] = count_discs

    # Add discs labels
    key_discs = 'discs-labels'
    label_list = np.arange(1,27).tolist() + [49, 50, 60]
    for num_label in label_list:
        if num_label in discs_labels:
            fprint_dict[filename][f'label_{str(num_label)}'] = True
        else:
            fprint_dict[filename][f'label_{str(num_label)}'] = False
    
    # Add dim and resolutions
    list_of_metrics = [X, Y, Z, nx, ny, nz, nt, px, py, pz, pt]
    list_of_keys = ['X', 'Y', 'Z', 'nx', 'ny', 'nz', 'nt', 'px', 'py', 'pz', 'pt']
    for key, metric in zip(list_of_keys, list_of_metrics):
        fprint_dict[filename][key] = metric

    return metrics_dict, fprint_dict


def save_violin(names, values, output_path, x_axis, y_axis):
    '''
    Create a violin plot
    :param names: String list of the names
    :param values: List of values associated with the names
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


def save_group_violins(name, values, output_path, x_axis, y_axis):
    '''
    Create a violin plot
    :param name: Dataset name
    :param values: List of metrics containing lists of values associated with the names
    :param output_path: Output path (string)
    :param x_axis: x-axis name
    :param y_axis: List of y-axis name corresponding to each metrics
    '''
    
    # Create plot 
    fig, axs = plt.subplots(3, len(values)//3 + 1, figsize=(1.8*len(values),11))

    fig.suptitle(f'{x_axis} : {name}', fontsize = 30)
    
    for idx_line, val in enumerate(values):
        # Set position of bar on X axis
        result_dict = {}
        result_dict['values'] = val
        result_dict['metrics'] = [y_axis[idx_line]]*len(val)
        
        result_df = pd.DataFrame(data=result_dict)

        # Make the plot
        sns.violinplot(ax=axs[idx_line//4, idx_line%4], x="metrics", y="values", data=result_df)
        axs[idx_line//4, idx_line%4].set(xticklabels=[])
        axs[idx_line//4, idx_line%4].set_ylabel("")
        axs[idx_line//4, idx_line%4].set_xlabel("")
        axs[idx_line//4, idx_line%4].set_title(y_axis[idx_line], fontsize=20)
    
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
    binwidth= 1/(1*len(names)) if len(names) > 1 else 1/3
    shrink = 1 if len(names) > 1 else 0.7
    plt.figure(figsize=(np.max(result_dict['values']), 8))
    sns.histplot(data=result_df, x="values", hue="names", multiple="dodge", binwidth=binwidth, shrink=shrink)
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
        # Regroup small values
        other_count = 0
        other_name_list = []
        for v, count in result_dict[name].items():
            if count <= math.ceil(0.004*len(values[i])):
                other_count += count
                other_name_list.append(v)
        for v in other_name_list:
            del result_dict[name][v]
        if other_name_list:
            result_dict[name]['other'] = other_count

    
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
        plt.xlabel(names[0], fontsize = 15)
        #plt.ylabel(y_axis, fontsize = 15)
    else:
        fig, axs = plt.subplots(1, len(names), figsize=(3*len(names),5))
        fig.suptitle(y_axis, fontsize = 8*len(names))

        for j, name in enumerate(result_dict.keys()):
            axs[j].pie(result_dict[name].values(), labels=result_dict[name].keys(), colors=palette_color, autopct=autopct_format(result_dict[names[j]].values()))
            axs[j].set_title(name)

        axs[0].set(ylabel=y_axis)
        
    # Save plot
    plt.savefig(output_path)

def convert_dict_to_float_list(dic):
    """
    This function converts dictionary with {str(value):int(nb_occurence)} to a list [float(value)]*nb_occurence
    """
    out_list = []
    for value, count in dic.items():
        out_list += [float(value)]*count
    return out_list

def convert_dict_to_list(dic):
    """
    This function converts dictionary with {str(value):int(nb_occurence)} to a list [str(value)]*nb_occurence
    """
    out_list = []
    for value, count in dic.items():
        out_list += [value]*count
    return out_list

def save_graphs(output_folder, metrics_dict, data_form='split'):
    '''
    Plot and save metrics into an output folder

    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    '''
    # Extract subjects and metrics
    data_name = np.array(list(metrics_dict.keys()))

    # Use violin plots
    # for metric, unit in zip(['nx', 'ny', 'nz', 'nt', 'px', 'py', 'pz', 'pt', 'X', 'Y', 'Z'], ['pixel', 'pixel', 'pixel', '', 'mm/pixel', 'mm/pixel', 'mm/pixel', '', 'mm', 'mm', 'mm']):
    #     out_path = os.path.join(output_folder, f'{metric}.png')
    #     metric_name = metric + ' ' + f'({unit})'
    #     save_violin(names=data_name, values=[convert_dict_to_float_list(metrics_dict[name][metric]) for name in data_name], output_path=out_path, x_axis=data_form, y_axis=metric_name)

    # Save violin plot in one fig
    for name in data_name:
        tot_values = []
        tot_names = []
        for metric, unit in zip(['nx', 'ny', 'nz', 'nt', 'px', 'py', 'pz', 'pt', 'X', 'Y', 'Z'], ['pixel', 'pixel', 'pixel', '', 'mm/pixel', 'mm/pixel', 'mm/pixel', '', 'mm', 'mm', 'mm']):
            tot_values.append(convert_dict_to_float_list(metrics_dict[name][metric]))
            tot_names.append(metric + ' ' + f'({unit})')
        out_path = os.path.join(output_folder, f'violin_stats.png')
        save_group_violins(name=name, values=tot_values, output_path=out_path, x_axis=data_form, y_axis=tot_names)

    # Use bar pie chart
    for metric in ['orientation', 'contrast']:
        out_path = os.path.join(output_folder, f'{metric}.png')
        save_pie(names=data_name, values=[convert_dict_to_list(metrics_dict[name][metric]) for name in data_name], output_path=out_path, x_axis=data_form, y_axis=metric)

    # Use bar graphs
    for metric in ['discs-labels']:
        out_path = os.path.join(output_folder, f'{metric}.png')
        save_hist(names=data_name, values=[convert_dict_to_float_list(metrics_dict[name][metric]) for name in data_name], output_path=out_path, x_axis=metric, y_axis='Count')

def mergedict(a,b):
    a.update(b)
    return a