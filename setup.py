from setuptools import find_packages, setup
from nocturne import __version__

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nocturne",
    packages=find_packages(exclude=["tests", "tests.*"]),
    version=__version__,
    description="Nocturne - utilities for Scalable Deep Learning with Spark, Horovod and Petastorm",
    author="Ivan Trusov",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Intended Audience :: Developers",
    ],
)
