"""
Read nnUNet training log file, extract epoch number and validation pseudo dice and plot them.
This is useful for comparing multi-class training (because nnUNet plots only the mean dice across classes).

Create venv and install dependencies
    python3 -m venv venv
    source ./venv/bin/activate
    pip install pandas plotly kaleido

Usage:
    python plot_nnunet_training_log.py -i <path_to_log_file>
    python plot_nnunet_training_log.py -i <path_to_log_file> -interactive-figure

Author: Jan Valosek
"""

import os
import re
import argparse

import pandas as pd
import plotly.express as px


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Plot epoch number vs Pseudo dice based on a nnUNet training log file.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        type=str,
        help='Path to the txt log file produced by nnUNet.'
             'Example: training_log_2024_1_22_11_09_18.txt'
    )
    parser.add_argument(
        '-interactive-figure',
        action='store_true',
        help='Show interactive figure. If not used, the figure will be only saved to a file.'
    )

    return parser


def extract_epoch_and_dice(log_file_path):
    """
    Extract fold number and epoch and pseudo dice from the log file.
    Args:
        log_file_path: Path to the log file.
    Returns:
        epoch_and_dice_data: List of dictionaries with keys 'epoch' and 'pseudo_dice'.
        fold_number: Fold number used for training.
    """
    fold_pattern = re.compile(r'Desired fold for training: (\d+)')
    epoch_pattern = re.compile(r'Epoch (\d+)')
    dice_pattern = re.compile(r'Pseudo dice \[([^,\]]+(?:, [^,\]]+)*)\]')

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    epoch_and_dice_data = []
    for line in lines:
        fold_match = fold_pattern.search(line)
        epoch_match = epoch_pattern.search(line)
        dice_match = dice_pattern.search(line)

        if fold_match:
            fold_number = int(fold_match.group(1))

        if epoch_match:
            epoch = int(epoch_match.group(1))
            epoch_and_dice_data.append({'epoch': epoch, 'pseudo_dice': None})

        elif dice_match:
            # Extracting the list using regular expression
            list_match = re.search(r'\[.*\]', dice_match.group())

            if list_match:
                extracted_list_str = list_match.group(0)
                # Replace 'nan' with the actual nan value
                extracted_list_str = extracted_list_str.replace('nan', 'float("nan")')
                # Evaluating the string representation of the list
                extracted_list = eval(extracted_list_str)
                epoch_and_dice_data[-1]['pseudo_dice'] = extracted_list

    # Check if fold_number is instance of int ("if not fold_number:" does not work for '0')
    if not isinstance(fold_number, int):
        fold_number = 'all'

    return epoch_and_dice_data, fold_number


def create_figure(df, log_file_path, fold_number, args):
    # Plotting using Plotly Express
    fig = px.line(df, x='epoch', y=df.columns[1:])
    # Update line width to 3
    fig.update_traces(line=dict(width=3))
    # Update the line color for 'pseudo_dice_mean' to black
    fig.update_traces(line=dict(color='black', width=5), selector=dict(name='pseudo_dice_mean'))
    # Fix the y-axis range to be between 0 and 1
    fig.update_yaxes(range=[-0.1, 1.1])
    # Add x-axis title
    fig.update_xaxes(title_text='Epoch')
    # Add y-axis title
    fig.update_yaxes(title_text='Validation Pseudo Dice')
    # Add title with fold number
    fig.update_layout(title=f'Fold {fold_number} -- Validation Pseudo Dice vs. Epoch')

    # Increase all font sizes
    fig.update_layout(font=dict(size=28))

    # Show the interactive plot
    if args.interactive_figure:
        fig.show()

    # Save plot to a file
    fname_fig = log_file_path.replace('.txt', '.png')
    fig.write_image(fname_fig, width=1920, height=1080)
    print(f'Saved plot to {fname_fig}')

    # Print the latest Dice for each class
    print(f'Latest Validation Pseudo Dice for each class: {df.iloc[-2, 1:-1].to_list()}')
    # Print the mean Dice across all classes
    print(f'Latest Mean Validation Pseudo Dice across all classes: {df.iloc[-2, -1]}')

    # SUPER RELEVANT discussion "validation dice" vs. "validation pseudo-dice":
    # https://github.com/ivadomed/utilities/pull/41#discussion_r1465360824
    # - "validation dice" is computed from validation samples (not used for training) on full images (not patches)
    # after the model training completely finish
    # - "validation pseudo-dice" is computed on randomly drawn patches (not full images) from the validation data at
    # the end of each epoch


def main():

    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Get absolute path to the log file
    log_file_path = os.path.abspath(os.path.expanduser(args.i))

    data, fold_number = extract_epoch_and_dice(log_file_path)

    # Convert data to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Create columns for each element in pseudo_dice
    # [:-1] is used to remove the last row, which is empty
    df_pseudo_dice = pd.DataFrame(df['pseudo_dice'].to_list()[:-1],
                                  columns=[f'validation_pseudo_dice_class_{i+1}'
                                           for i in range(len(df['pseudo_dice'].iloc[0]))])

    # Concatenate the new DataFrame with the original DataFrame
    df = pd.concat([df, df_pseudo_dice], axis=1).drop('pseudo_dice', axis=1)

    # Compute mean of pseudo dice across all classes
    df['validation_pseudo_dice_mean'] = df.iloc[:, 1:].mean(axis=1)

    # Create figure
    create_figure(df, log_file_path, fold_number, args)


if __name__ == "__main__":
    main()
