#######################################################################
#
# Tests for the `dataset_conversion/test_convert_bids_to_nnUNetV2.py` script
#
# RUN BY:
#   python -m unittest tests/test_convert_bids_to_nnUNetV2.py
#######################################################################

import unittest
from unittest.mock import patch
from dataset_conversion.convert_bids_to_nnUNetV2 import convert_subject, get_parser


class TestConvertBidsToNnunet(unittest.TestCase):
    """
    Test the conversion of BIDS dataset to nnUNetV2 dataset with data_type = "anat"
    """

    def setUp(self):
        self.root = "/path/to/bids"
        self.subject = "sub-001"
        self.contrast = "T2w"
        self.label_suffix = "lesion-manual"
        self.data_type = "anat"
        self.path_out_images = "/path/to/nnunet/imagesTr"
        self.path_out_labels = "/path/to/nnunet/labelsTr"
        self.counter = 0
        self.list_images = []
        self.list_labels = []
        self.is_ses = False
        self.copy = False
        self.DS_name = "MyDataset"
        self.session = None
        self.channel = 0

    @patch('os.path.exists')
    @patch('os.symlink')
    @patch('shutil.copy2')
    def test_convert_subject(self, mock_copy, mock_symlink, mock_exists):
        # Setup mock responses
        mock_exists.side_effect = lambda x: True  # Simulate that all files exist

        # Execute the function
        result_images, result_labels = convert_subject(
            self.root, self.subject, self.channel, self.contrast, self.label_suffix,
            self.data_type, self.path_out_images, self.path_out_labels, self.counter,
            self.list_images, self.list_labels, self.is_ses, self.copy, self.DS_name,
            self.session
        )

        # Assert conditions
        self.assertEqual(len(result_images), 1)
        self.assertEqual(len(result_labels), 1)
        if self.copy:
            mock_copy.assert_called()
        else:
            mock_symlink.assert_called()

    def test_argument_parsing(self):
        parser = get_parser()
        args = parser.parse_args([
            '--path-data', '/path/to/bids', '--path-out', '/path/to/nnunet',
            '--contrast', 'T2w', '--label-suffix', 'lesion-manual',
            '--data-type', 'anat', '--dataset-name', 'MyDataset',
            '--dataset-number', '501', '--split', '0.8', '0.2', '--seed', '99', '--copy', 'True'
        ])
        self.assertEqual(args.path_data, '/path/to/bids')
        self.assertEqual(args.path_out, '/path/to/nnunet')
        self.assertEqual(args.contrast, ['T2w'])
        self.assertEqual(args.label_suffix, 'lesion-manual')
        self.assertEqual(args.data_type, 'anat')
        self.assertEqual(args.dataset_name, 'MyDataset')
        self.assertEqual(args.dataset_number, 501)
        self.assertEqual(args.split, [0.8, 0.2])
        self.assertEqual(args.seed, 99)
        self.assertEqual(args.copy, True)


class TestConvertBidsToNnunetFuncDataType(unittest.TestCase):
    """
    Test the conversion of BIDS dataset to nnUNetV2 dataset with data_type = "func"
    """

    def setUp(self):
        # Setup common test data for the "func" data type scenario
        self.root = "/path/to/bids"
        self.subject = "sub-001"
        self.contrast = "bold"
        self.label_suffix = "task-rest"
        self.data_type = "func"
        self.path_out_images = "/path/to/nnunet/imagesTr"
        self.path_out_labels = "/path/to/nnunet/labelsTr"
        self.counter = 0
        self.list_images = []
        self.list_labels = []
        self.is_ses = False
        self.copy = False
        self.DS_name = "MyDataset"
        self.session = None
        self.channel = 0

    @patch('os.path.exists')
    @patch('os.symlink')
    @patch('shutil.copy2')
    @patch('os.listdir')
    def test_convert_subject_func(self, mock_listdir, mock_copy, mock_symlink, mock_exists):
        # Mock the os.listdir to return files simulating a "func" directory structure
        mock_listdir.side_effect = lambda x: ["sub-001_task-rest_bold.nii.gz"]
        # Mock os.path.exists to simulate that all necessary files exist
        mock_exists.side_effect = lambda x: True

        # Execute the function with "func" data type
        result_images, result_labels = convert_subject(
          self.root, self.subject, self.channel, self.contrast, self.label_suffix,
          self.data_type, self.path_out_images, self.path_out_labels, self.counter,
          self.list_images, self.list_labels, self.is_ses, self.copy, self.DS_name,
          self.session
        )

        # Assert conditions specific to "func" data type processing
        self.assertEqual(len(result_images), 1, "Should have added one image path for 'func' data type")
        self.assertEqual(len(result_labels), 1, "Should have added one label path for 'func' data type")

        expected_image_path = f"{self.path_out_images}/{self.DS_name}-sub-001_{self.counter:03d}_{self.channel:04d}.nii.gz"
        expected_label_path = f"{self.path_out_labels}/{self.DS_name}-sub-001_{self.counter:03d}.nii.gz"

        self.assertIn(expected_image_path, result_images, "The image path for 'func' data type is not as expected")
        self.assertIn(expected_label_path, result_labels, "The label path for 'func' data type is not as expected")

        if self.copy:
          mock_copy.assert_called()
        else:
          mock_symlink.assert_called()


if __name__ == '__main__':
    unittest.main()
