# This purpose of thie file is to make our whole ML project a package, so other people can install it easily.

from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("/n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='CO2 Emission Prediction',
    version='0.1.0',
    author='Amanat Kazmi',
    # Automatically find all packages in the project directory where there is an __init__.py file.
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
