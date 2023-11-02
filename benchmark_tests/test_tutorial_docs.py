import re


def test_tutorial_docs():
    with open("./docs/tutorials.rst", "r") as fi:
        lines = fi.read().splitlines()

    # find start lines
    idx_ini = [i for i, val, in enumerate(lines) if val == ".. code-block:: python"]

    data = []
    for ini in idx_ini:
        # find end line
        for val in lines[ini:]:
            if len(val.lstrip()) > 0 and re.search('\S', val).start() >= 4:
                data.append(val.lstrip())

    exec("\n".join(data))
