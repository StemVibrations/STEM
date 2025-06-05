from .__version__ import __version__
import subprocess
import importlib.metadata


def install_external_dependency():
    """
    Install external dependencies for STEM
    """
    if not check_package_version('StemKratos', '1.2.1'):
        subprocess.run(['pip', 'install', "git+https://github.com/StemVibrations/StemKratos@v1.2.1"])


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
        package_version = importlib.metadata.version(package_name)

        # Compare the installed version with the target version
        return package_version == target_version

    except importlib.metadata.PackageNotFoundError:
        return False


install_external_dependency()
