import re
from typing import List
import shutil


def find_first_greater(my_list: List[float], value: float) -> float:
    """
    Find the first value greater than the given value

    Args:
        - my_list (list): list of values
        - value (float): value to compare

    Returns:
        - float: first value greater than the given value
    """
    for item in my_list:
        if item > value:
            return item


def read_tutorial(rst_file: str, name: str) -> List[str]:
    """
    Read the code from the rst file

    Args:
        - rst_file (str): path to the rst file
        - name (str): name of the tutorial

    Returns:
        - List[str]: list of strings with the code
    """

    with open(rst_file, "r") as fi:
        lines = fi.read().splitlines()

    # find start line of tutorial and end of tutorial
    idx_ini = [i for i, val in enumerate(lines) if name in val][0]
    idx_end = [i for i, val in enumerate(lines) if "_tutorial" in val]
    idx_end = find_first_greater(idx_end, idx_ini)

    tutorial = lines[idx_ini: idx_end]

    # find start of python code in tutorial, by checking for: '.. code-block:: python'
    idx_ini = [i for i, val, in enumerate(tutorial) if val == ".. code-block:: python"]
    idx_ini.append(idx_end)

    data = []
    # for each code block
    for i in range(len(idx_ini) - 1):
        # find end line
        for val in tutorial[idx_ini[i]:idx_ini[i+1]]:
            # find the code inside the code block. the code should have at least 4 spaces and not be empty
            if len(val.lstrip()) > 0 and re.search('\S', val).start() >= 4:
                data.append(val.lstrip())

    return data

def test_tutorial_1():
    """Test the code in tutorial 1"""
    name = "_tutorial1"
    tutorial_file = "./docs/tutorials.rst"

    data = read_tutorial(tutorial_file, name)
    exec("\n".join(data))
    shutil.rmtree("line_load")


def test_tutorial_2():
    """Test the code in tutorial 2"""
    name = "_tutorial2"
    tutorial_file = "./docs/tutorials.rst"

    data = read_tutorial(tutorial_file, name)
    exec("\n".join(data))
    shutil.rmtree("moving_load")

def test_tutorial_3():
    """Test the code in tutorial 3"""
    name = "_tutorial3"
    tutorial_file = "./docs/tutorials.rst"

    data = read_tutorial(tutorial_file, name)
    exec("\n".join(data))
    shutil.rmtree("uvec_train_model")
