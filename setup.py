from setuptools import find_packages, setup
from dbx_scalable_dl import __version__

INSTALL_REQUIRES = [
    "tensorflow==2.6.0",
    "tensorflow-recommenders==0.6.0",
    "petastorm==0.11.2",
    "horovod==0.22.1",
]

setup(
    name="dbx_scalable_dl",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="",
    author="",
)
