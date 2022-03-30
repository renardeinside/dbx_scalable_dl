from setuptools import find_packages, setup
from nocturne import __version__

INSTALL_REQUIRES = ["dunamai"]

# setup_requires is set externally via pyproject.toml

setup(
    name="nocturne",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=INSTALL_REQUIRES,
    version=__version__,
    description="",
    author="",
)
