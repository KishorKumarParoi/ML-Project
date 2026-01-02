from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="ML-Project",
    version="0.1.0",
    author="Kishor Kumar Paroi",
    author_email="kishor.ruet.cse@gmail.com",
    description="A machine learning project setup",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.6",
)

# python3 -m venv venv
# source venv/bin/activate
# python -m pip install --upgrade pip setuptools wheel
# pip install -r requirements.txt
