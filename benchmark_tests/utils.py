import os

def assert_files_equal(exact_folder, test_folder):

    files = os.listdir(exact_folder)

    for file in files:

        with open(os.path.join(exact_folder, file)) as fi:
            exact = fi.read()
        with open(os.path.join(test_folder, file)) as fi:
            test = fi.read()

        if exact == test:
            return True

    return False