from setuptools import find_packages, setup
from nocturne import __version__

setup(
    name="nocturne",
    packages=find_packages(exclude=["tests", "tests.*"]),
    version=__version__,
    description="",
    author="Ivan Trusov",
)
