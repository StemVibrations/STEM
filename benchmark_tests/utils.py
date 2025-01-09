import os
import re

import numpy as np
import numpy.testing as npt


def assert_floats_in_files_almost_equal(exact_file_name: str, test_file_name: str, decimal: int = 9):
    r"""
    Compares two files and checks if all floats in the files are almost equal.

    Args:
        - exact_file_name (str): The exact file name.
        - test_file_name (str): The test file name.
        - decimal (float): Desired precision, default is 9.

    """

    with open(exact_file_name) as fi:
        exact = fi.read()
    with open(test_file_name) as fi:
        test = fi.read()

    # get all floats from files
    all_floats_exact = np.array(re.findall(r'[\d]*[.][\d]+', exact), float)
    all_floats_test = np.array(re.findall(r'[\d]*[.][\d]+', test), float)

    # check if number of floats is equal
    assert len(all_floats_exact) == len(all_floats_test)
    npt.assert_almost_equal(all_floats_exact, all_floats_test, decimal=decimal)


def assert_floats_in_directories_almost_equal(exact_folder: str, test_folder: str, decimal: int = 9):
    r"""
    Compares two folders containing files and checks if all floats in the files are almost equal.

    Args:
        - exact_folder (str): The folder containing the exact files.
        - test_folder (str): The folder containing the test files.
        - decimal (float): Desired precision, default is 9.

    """

    # reads all files in directory
    files = os.listdir(exact_folder)

    # checks if all files in exact_folder are in test_folder
    for file in files:
        exact_file_name = os.path.join(exact_folder, file)
        test_file_name = os.path.join(test_folder, file)

        assert_floats_in_files_almost_equal(exact_file_name, test_file_name, decimal)


def assert_files_equal(exact_folder: str, test_folder: str) -> bool:
    r"""
    Compares two folders containing files and returns True if all files are equal, False otherwise.

    Args:
        - exact_folder (str): The folder containing the exact files.
        - test_folder (str): The folder containing the test files.

    Returns:
        - bool: True if all files are equal, False otherwise.
    """

    # reads all files in directory
    expected_files = os.listdir(exact_folder)
    calculated_files = os.listdir(exact_folder)
    assert len(expected_files) == len(calculated_files)

    # checks if all files in exact_folder are in test_folder
    for file in expected_files:
        with open(os.path.join(exact_folder, file)) as fi:
            exact = fi.read()
        with open(os.path.join(test_folder, file)) as fi:
            test = fi.read()

        # check if files are equal
        if exact == test:
            return True

    return False
