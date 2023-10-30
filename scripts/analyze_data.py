import os
import argparse
import json
from bids import BIDSLayout
import glob

from utils import get_img_path_from_mask_path, edit_metric_dict, save_graphs, change_mask_suffix

def run_analysis(args):
    """
    Run analysis on a config file
    """

    disc_label_suffix = '_labels-disc-manual'
    short_suffix_label = '_label'

    if args.config:
        data_type = 'split'
        # Read json file and create a dictionary
        with open(args.config, "r") as file:
            config_data = json.load(file)

        # Check if labels are specified
        if config_data['TYPE'] != 'LABEL':
            raise ValueError("Pease specify LABEL paths in config")
        
    elif args.paths_to_bids:
        data_type = 'dataset'
        # layout = BIDSLayout(args.paths_to_bids, derivatives=["derivatives/"])
        # tasks = layout.get_tasks()
        # layout.get(scope='derivatives', return_type='file')
        config_data = {}
        for path_bids in args.paths_to_bids:
            config_data[os.path.basename(path_bids)] = glob.glob(path_bids + "/**/*" + short_suffix_label + "*.nii.gz", recursive=True)
    else:
        raise ValueError(f"Need to specify either args.paths_to_bids or args.config !")


    # Initialize metrics dictionary``
    metrics_dict = dict()

    missing_data = []
    # Extract information from the data
    for key in config_data.keys():
        metrics_dict[key] = dict()
        for path in config_data[key]:
            img_path = get_img_path_from_mask_path(path)
            mask_path = path

            # Extract field of view information thanks to discs labels
            if short_suffix_label in mask_path:
                discs_mask_path = mask_path
            else:
                discs_mask_path = change_mask_suffix(mask_path, new_suffix=disc_label_suffix)
            
            # Extract data
            if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(discs_mask_path):
                metrics_dict[key] = edit_metric_dict(metrics_dict[key], img_path, mask_path, discs_mask_path)
            else:
                missing_data.append(img_path)
    
    print("missing files:\n" + '\n'.join(missing_data))

    # Plot data informations            
    save_graphs(output_folder='results', metrics_dict=metrics_dict, data_type=data_type)



                



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse config file')
    
    ## Parameters
    parser.add_argument('--paths-to-bids', default='', nargs='+',
                        help='Paths to BIDS compliant datasets')
    parser.add_argument('--config', default='',
                        help='Path to JSON config file that contains all the training splits')
    parser.add_argument('--split', default='ALL', choices=('TRAINING', 'VALIDATION', 'TESTING', 'ALL'),
                        help='Split of the data that will be analysed (default="ALL")')
    
    # Start analysis
    run_analysis(parser.parse_args())