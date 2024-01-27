# based on https://packaging.python.org/en/latest/tutorials/packaging-projects/

# remove old build
rm -rf ./dist
rm -rf ./build

# activate venv
source ./venv/bin/activate

# install build tools
pip install --upgrade build
pip install --upgrade twine

# build the package
python -m build

# upload to pypi
python -m twine upload dist/* --verbose