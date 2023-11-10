import os
import argparse
import json
import glob
from progress.bar import Bar
import csv

from utils import get_img_path_from_mask_path, get_mask_path_from_img_path, edit_metric_dict, save_graphs, change_mask_suffix, get_deriv_sub_from_img_path, str_to_float_list, str_to_str_list, mergedict

def run_analysis(args):
    """
    Run analysis on a config file
    """

    short_suffix_disc = '_label'
    short_suffix_seg = '_seg'
    derivatives_folder = 'derivatives'
    output_folder = 'results'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.config:
        data_form = 'split'
        # Read json file and create a dictionary
        with open(args.config, "r") as file:
            config_data = json.load(file)
        
        if config_data['TYPE'] == 'LABEL':
            isImage = False
        elif config_data['TYPE'] == 'IMAGE':
            isImage = True
        else:
            raise ValueError(f'config with unknown TYPE {config_data['TYPE']}')
        
        # Remove keys that are not lists of paths
        keys = list(config_data.keys())
        for key in keys:
            if key not in ['TRAINING', 'VALIDATION', 'TESTING']:
                del config_data[key]
        
    elif args.paths_to_bids:
        data_form = 'dataset'
        config_data = {}
        for path_bids in args.paths_to_bids:
            files = glob.glob(path_bids + "/**/" + "*.nii.gz", recursive=True) # Get all niftii files
            config_data[os.path.basename(os.path.normpath(path_bids))] = [f for f in files if derivatives_folder not in f] # Remove masks from derivatives folder
        isImage = True
    
    elif args.paths_to_csv:
        data_form = 'dataset'
        config_data = {}
    else:
        raise ValueError(f"Need to specify either args.paths_to_bids, args.config or args.paths_to_csv !")

    # Initialize metrics dictionary
    metrics_dict = dict()

    if args.paths_to_csv: 
        for path_csv in args.paths_to_csv:
            dataset_name = os.path.basename(path_csv).split('_')[-1].split('.csv')[0]
            metrics_dict[dataset_name] = {}
            with open(path_csv) as csv_file:
                reader = csv.reader(csv_file)
                for k, v in dict(reader).items():
                    metric = k.split('_')
                    if len(metric) == 2:
                        metric_name, metric_value = metric
                        if metric_name not in metrics_dict[dataset_name].keys():
                            metrics_dict[dataset_name][metric_name] = {metric_value:int(v)}
                        else:
                            metrics_dict[dataset_name][metric_name][metric_value] = int(v)
                    else:
                        if k.startswith('mismatch'):
                            metrics_dict[dataset_name][k] = int(v)
                        else:
                            metrics_dict[dataset_name][k] =  str_to_str_list(v)

    # Initialize data finguerprint
    fprint_dict = dict()

    if config_data.keys():
        missing_data = []
        # Extract information from the data
        for key in config_data.keys():
            metrics_dict[key] = dict()
            fprint_dict[key] = dict()

            # Init progression bar
            bar = Bar(f'Analyze data {key} ', max=len(config_data[key]))

            for path in config_data[key]:
                if isImage:
                    img_path = path # str
                    deriv_sub_folders = get_deriv_sub_from_img_path(img_path=img_path, derivatives_folder=derivatives_folder) # list of str
                    seg_paths = get_mask_path_from_img_path(img_path, short_suffix=short_suffix_seg, deriv_sub_folders=deriv_sub_folders) # list of str
                    discs_paths = get_mask_path_from_img_path(img_path, short_suffix=short_suffix_disc, deriv_sub_folders=deriv_sub_folders, counterexample=['compression', 'SC_mask', 'seg']) # list of str
                else:
                    img_path = get_img_path_from_mask_path(path, derivatives_folder=derivatives_folder)
                    deriv_sub_folders = [os.path.dirname(path)]
                    # Extract field of view information thanks to discs labels
                    if short_suffix_disc in path:
                        discs_paths = [path]
                        seg_paths = [change_mask_suffix(discs_paths, short_suffix=short_suffix_seg)]
                    elif short_suffix_seg in path:
                        seg_paths = [path]
                        discs_paths = [change_mask_suffix(seg_paths, short_suffix=short_suffix_disc)]
                    else:
                        seg_paths = [change_mask_suffix(path, short_suffix=short_suffix_seg)]
                        discs_paths = [change_mask_suffix(path, short_suffix=short_suffix_disc)]

                # Extract data
                if os.path.exists(img_path):
                    metrics_dict[key], fprint_dict[key] = edit_metric_dict(metrics_dict[key], fprint_dict[key], img_path, seg_paths, discs_paths, deriv_sub_folders)
                else:
                    missing_data.append(img_path)
        
                # Plot progress
                bar.suffix  = f'{config_data[key].index(path)+1}/{len(config_data[key])}'
                bar.next()
            bar.finish()

            # Store csv with computed metrics
            if args.create_csv:
                # Based on https://stackoverflow.com/questions/8685809/writing-a-dictionary-to-a-csv-file-with-one-line-for-every-key-value
                out_csv_folder = os.path.join(output_folder, 'files')
                if not os.path.exists(out_csv_folder):
                    os.makedirs(out_csv_folder)
                csv_path_sum = os.path.join(out_csv_folder, f'computed_metrics_{key}.csv')
                with open(csv_path_sum, 'w') as csv_file:  
                    writer = csv.writer(csv_file)
                    for metric_name, metric in sorted(metrics_dict[key].items()):
                        if isinstance(metric,dict):
                            for metric_value, count in sorted(metric.items()):
                                k = f'{metric_name}_{metric_value}'
                                writer.writerow([k, count])
                        else:
                            writer.writerow([metric_name, metric])
                
                # Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark 
                csv_path_fprint = os.path.join(out_csv_folder, f'fprint_{key}.csv')
                sub_list = [sub for sub in fprint_dict[key].keys() if sub.startswith('sub')]
                fields = ['subject'] + [k for k in fprint_dict[key][sub_list[0]].keys()]
                with open(csv_path_fprint, 'w') as f:  
                    w = csv.DictWriter(f, fields)
                    w.writeheader()
                    for k, v in fprint_dict[key].items():
                        w.writerow(mergedict({'subject': k},v))

        
        if missing_data:
            print("missing files:\n" + '\n'.join(missing_data))

    # Plot data informations            
    save_graphs(output_folder=output_folder, metrics_dict=metrics_dict, data_form=data_form)



                



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse config file')
    
    ## Parameters
    parser.add_argument('--paths-to-bids', default='', nargs='+',
                        help='Paths to BIDS compliant datasets (You can add multiple paths using spaces)')
    parser.add_argument('--config', default='',
                        help='Path to JSON config file that contains all the training splits')
    parser.add_argument('--paths-to-csv', default='', nargs='+',
                        help='Paths to csv files with already computed metrics (You can add multiple paths using spaces)')
    parser.add_argument('--split', default='ALL', choices=('TRAINING', 'VALIDATION', 'TESTING', 'ALL'),
                        help='Split of the data that will be analysed (default="ALL")')
    parser.add_argument('--create-csv', default=True,
                        help='Store computed metrics using a csv file in results/files (default=True)')
    
    # Start analysis
    run_analysis(parser.parse_args())