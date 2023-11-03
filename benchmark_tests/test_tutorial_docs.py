import re
import shutil
import pytest

@pytest.fixture(scope="function")
def teardown(request):
    """
    Delete the folder at the end of the test
    """
    folder_name = request.param
    yield
    shutil.rmtree(folder_name)

def find_first_greater(my_list, value):
    """
    Find the first value greater than the given value

    Args:
        my_list (list): list of values
        value (float): value to compare
    """
    for item in my_list:
        if item > value:
            return item
    return None

def read_tutorial(rst_file, name):
    """
    Read the code from the rst file

    Args:
        rst_file (str): path to the rst file
        name (str): name of the tutorial
    """

    with open(rst_file, "r") as fi:
        lines = fi.read().splitlines()

    # find start line of tutorial and end of tutorial
    idx_ini = [i for i, val in enumerate(lines) if name in val][0]
    idx_end = [i for i, val in enumerate(lines) if "_tutorial" in val]
    idx_end = find_first_greater(idx_end, idx_ini)

    tutorial = lines[idx_ini: idx_end]

    # find start of python code in tutorial
    idx_ini = [i for i, val, in enumerate(tutorial) if val == ".. code-block:: python"]
    idx_ini.append(idx_end)

    data = []
    for i in range(len(idx_ini) - 1):
        # find end line
        for val in tutorial[idx_ini[i]:idx_ini[i+1]]:
            if len(val.lstrip()) > 0 and re.search('\S', val).start() >= 4:
                data.append(val.lstrip())

    return data

@pytest.mark.parametrize('teardown', ['./line_load'], indirect=True)
def test_tutorial_1(teardown):
    """Test the code in tutorial 1"""
    name = "_tutorial1"
    tutorial_file = "./docs/tutorials.rst"

    data = read_tutorial(tutorial_file, name)
    exec("\n".join(data))

@pytest.mark.parametrize('teardown', ['./moving_load'], indirect=True)
def test_tutorial_2(teardown):
    """Test the code in tutorial 2"""
    name = "_tutorial2"
    tutorial_file = "./docs/tutorials.rst"

    data = read_tutorial(tutorial_file, name)
    exec("\n".join(data))
