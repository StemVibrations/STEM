# Generate documentation from docstrings using Sphinx

To generate documentation from docstrings using Sphinx with Python, you need to follow these steps:

## Step 1: Install Sphinx
Ensure that you have Sphinx installed. You can install it using pip:

```bash
pip install sphinx
```

## Step 2: Initialize the documentation project
Create a new directory for your documentation project and navigate to it. Then, run the following command to initialize the documentation project:

```bash
sphinx-quickstart
```

This command will prompt you with a series of questions to set up the basic configuration for your documentation project. You can press Enter to accept the default options or provide your own values.

## Step 3: Configure Sphinx
Once the project is initialized, you'll have a conf.py file in your project directory. Open this file in a text editor and locate the extensions list. Uncomment the line that includes 'sphinx.ext.autodoc' to enable the autodoc extension.

## Step 4: Write your Python code with docstrings
In your Python code, ensure that you have docstrings written for the classes, functions, or modules that you want to document. Here's an example:

```python
def add(a, b):
    """
    Adds two numbers.

    :param a: First number
    :param b: Second number
    :return: Sum of the two numbers
    """
    return a + b
```

## Step 5: Generate the documentation
To generate the documentation from the docstrings, run the following command in your project directory:

```bash
sphinx-apidoc -o docs ./stem
```


This command will generate .rst files in the docs/ directory based on your Python code and its docstrings.

## Step 6: Build the HTML documentation
Now, you can build the HTML documentation using the following command:

```bash
sphinx-build -b html ./docs/ ./docs/build/
```
This command will create the HTML documentation in the docs/ directory.

## Step 7: View the documentation
Open the generated HTML documentation files in your web browser to view the documentation. You can find the main index.html file in the ./docs/build/ directory.


# Publish to PyPI
To publish your package to PyPI, you need to run this [file](./build_pack.sh).

In order to get it working with PyPI you need to create a file called ~/.pypirc with the following content:

```bash

    [distutils]
    index-servers =
    pypi
    testpypi

    [pypi]
    username = __token__
    password = <TOKEN>

    [testpypi]
    username = __token__
    password = <TOKEN>
    ```
