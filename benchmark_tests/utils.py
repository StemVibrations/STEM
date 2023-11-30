import os

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
    files = os.listdir(exact_folder)

    # checks if all files in exact_folder are in test_folder
    for file in files:
        with open(os.path.join(exact_folder, file)) as fi:
            exact = fi.read()
        with open(os.path.join(test_folder, file)) as fi:
            test = fi.read()

        # check if files are equal
        if exact == test:
            return True

    return False