from .__version__ import __version__
import subprocess
import pkg_resources


def install_external_dependency():
    """
    Install external dependencies for STEM
    """
    # check if gmsh utils is installed and it is version 1.0
    if not check_package_version('gmsh_utils', '1.0'):
        subprocess.run(['pip', 'install', "git+https://github.com/StemVibrations/gmsh_utils@v1.0"])
    # check if RandomFields is installed and it is version 1.0
    if not check_package_version('random_fields', '1.0'):
        subprocess.run(['pip', 'install', "git+https://github.com/StemVibrations/RandomFields@v1.0"])
    # check if RandFields is installed and it is version 1.0
    if not check_package_version('StemKratos', '1.0'):
        subprocess.run(['pip', 'install', "git+https://github.com/StemVibrations/StemKratos@v1.0"])


def check_package_version(package_name: str, target_version: str) -> bool:
    """
    Check if a package is installed and if the installed version matches the target version

    Args:
        - package_name (str): Name of the package to check
        - target_version (str): Version of the package to check

    Returns:
        bool: True if the package is installed and the version matches the target version, False otherwise
    """
    try:
        # Use `require` to check if the package is installed and get its version
        package_version = pkg_resources.require(package_name)[0].version

        # Compare the installed version with the target version
        if package_version == target_version:
            return True
        else:
            return False
    except Exception:
        return False


install_external_dependency()
