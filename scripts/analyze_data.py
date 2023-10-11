import os
import argparse
import json

from utils import get_img_path_from_mask_path, edit_metric_dict, save_graphs

def run_analysis(args):
    """
    Run analysis on a config file
    """

    disc_label_suffix = '_labels-disc-manual'

    # Check if labels are specified
    if config_data['TYPE'] != 'LABEL':
        raise ValueError("Pease specify l")

    # Read json file and create a dictionary
    with open(args.config, "r") as file:
        config_data = json.load(file)

    # Check analysis split
    if args.split == 'ALL':
        data_split = ['TRAINING', 'VALIDATION', 'TESTING']
    elif args.split in ['TRAINING', 'VALIDATION', 'TESTING']:
        data_split = [args.split]
    else:
        raise ValueError(f"Invalid args.split: {args.split}")

    # Initialize metrics dictionary
    metrics_dict = dict()

    # Extract information from the data
    for split in data_split:
        metrics_dict[split] = dict()
        if config_data[split]:
            for path in config_data[split]:
                img_path = get_img_path_from_mask_path(path)
                mask_path = path
                
                # Extract data
                metrics_dict[split] = edit_metric_dict(metrics_dict[split], img_path, mask_path, disc_label_suffix=disc_label_suffix)

    # Plot data informations            
    save_graphs(output_folder='results', metrics_dict=metrics_dict)



                



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse config file')
    
    ## Parameters
    parser.add_argument('--config', required=True,
                        help='Path to JSON config file that contains all the training splits (Required)')
    parser.add_argument('--split', default='ALL', choices=('TRAINING', 'VALIDATION', 'TESTING', 'ALL'),
                        help='Split of the data that will be analysed (default="ALL")')
    
    # Start analysis
    run_analysis(parser.parse_args())