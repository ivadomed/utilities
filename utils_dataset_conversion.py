#!/usr/bin/env python
# -*- coding: utf-8
#
# Collection of useful functions used by other scripts
#
# Authors: Jan Valosek, Sandrine Bédard, Julien Cohen-Adad, Naga Karthik
#

import os
import re
import logging
import sys
import textwrap
import argparse
import subprocess
import shutil
import yaml

import numpy as np
import nibabel as nib


# BIDS utility tool
def fetch_subject_info(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subjectID: subject ID (e.g., sub-001)
    :return: sessionID: session ID (e.g., ses-01)
    :return: filename: nii filename (e.g., sub-001_ses-01_T1w.nii.gz)
    """

    _, filename = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    subject = re.search('sub-(.*?)[_/]', filename_path)
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash
    
    session = re.search('ses-(.*?)[_/]', filename_path)
    sessionID = session.group(0)[:-1] if session else ""    # [:-1] removes the last underscore or slash

    # search for acq- suffix in filename
    contrast_suffix = re.search('acq-(.*?)[./]', filename_path)
    if not contrast_suffix:
        contrast_suffix = re.search(f"(?<={subjectID}_)(.*)(?=nii.gz)", filename_path)
        
    contrast_suffixID = contrast_suffix.group(0)[:-1] if contrast_suffix else ""
    
    datatype = 'dwi' if 'dwi' in filename_path else 'anat'  # Return contrast (dwi or anat)
    # REGEX explanation
    # \d - digit
    # \d? - no or one occurrence of digit
    # *? - match the previous element as few times as possible (zero or more times)

    return subjectID, sessionID, contrast_suffixID, datatype, filename


class SmartFormatter(argparse.HelpFormatter):
    """
    Custom formatter that inherits from HelpFormatter, which adjusts the default width to the current Terminal size,
    and that gives the possibility to bypass argparse's default formatting by adding "R|" at the beginning of the text.
    Inspired from: https://pythonhosted.org/skaff/_modules/skaff/cli.html
    """
    def __init__(self, *args, **kw):
        self._add_defaults = None
        super(SmartFormatter, self).__init__(*args, **kw)
        # Update _width to match Terminal width
        try:
            self._width = shutil.get_terminal_size()[0]
        except (KeyError, ValueError):
            logging.warning('Not able to fetch Terminal width. Using default: %s'.format(self._width))

    # this is the RawTextHelpFormatter._fill_text
    def _fill_text(self, text, width, indent):
        # print("splot",text)
        if text.startswith('R|'):
            paragraphs = text[2:].splitlines()
            rebroken = [textwrap.wrap(tpar, width) for tpar in paragraphs]
            rebrokenstr = []
            for tlinearr in rebroken:
                if (len(tlinearr) == 0):
                    rebrokenstr.append("")
                else:
                    for tlinepiece in tlinearr:
                        rebrokenstr.append(tlinepiece)
            return '\n'.join(rebrokenstr)  # (argparse._textwrap.wrap(text[2:], width))
        return argparse.RawDescriptionHelpFormatter._fill_text(self, text, width, indent)
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            lines = text[2:].splitlines()
            while lines[0] == '':  # Discard empty start lines
                lines = lines[1:]
            offsets = [re.match("^[ \t]*", l).group(0) for l in lines]
            wrapped = []
            for i in range(len(lines)):
                li = lines[i]
                if len(li) > 0:
                    o = offsets[i]
                    ol = len(o)
                    init_wrap = textwrap.fill(li, width).splitlines()
                    first = init_wrap[0]
                    rest = "\n".join(init_wrap[1:])
                    rest_wrap = textwrap.fill(rest, width - ol).splitlines()
                    offset_lines = [o + wl for wl in rest_wrap]
                    wrapped = wrapped + [first] + offset_lines
                else:
                    wrapped = wrapped + [li]
            return wrapped
        return argparse.HelpFormatter._split_lines(self, text, width)


def splitext(fname):
        """
        Split a fname (folder/file + ext) into a folder/file and extension.

        Note: for .nii.gz the extension is understandably .nii.gz, not .gz
        (``os.path.splitext()`` would want to do the latter, hence the special case).
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
    Add suffix between end of file name and extension.

    :param fname: absolute or relative file name. Example: t2.nii
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii

    Examples:

    - add_suffix(t2.nii, _mean) -> t2_mean.nii
    - add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem + suffix + ext)


def remove_suffix(fname, suffix):
    """
    Remove suffix between end of file name and extension.

    :param fname: absolute or relative file name with suffix. Example: t2_mean.nii
    :param suffix: suffix. Example: _mean
    :return: file name without suffix. Example: t2.nii

    Examples:

    - remove_suffix(t2_mean.nii, _mean) -> t2.nii
    - remove_suffix(t2a.nii.gz, a) -> t2.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem.replace(suffix, '') + ext)


def get_full_path(path):
    """
    Return full path. If ~ is passed, expand it to home directory.
    :param path: str: Input path
    :return: str: Full path
    """
    return os.path.abspath(os.path.expanduser(path))


def check_files_exist(dict_files, path_data):
    """
    Check if all files listed in the input dictionary exist
    :param dict_files:
    :param path_data: folder where BIDS dataset is located
    :return:
    """
    missing_files = []
    for task, files in dict_files.items():
        # Do no check if key is empty or if regex is used
        if files is not None and '*' in files:
            for file in files:
                subject, ses, filename, contrast = fetch_subject_info(file)
                fname = os.path.join(path_data, subject, ses, contrast, filename)
                if not os.path.exists(fname):
                    missing_files.append(fname)
    if missing_files:
        logging.error("The following files are missing: \n{}. \nPlease check that the files listed "
                        "in the yaml file and the input path are correct.".format(missing_files))


