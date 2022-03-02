from setuptools import find_packages, setup
from dbx_scalable_dl import __version__

INSTALL_REQUIRES = ["tensorflow-recommenders==0.6.0"]

setup(
    name="dbx_scalable_dl",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    install_requires=INSTALL_REQUIRES,
    version=__version__,
    description="",
    author="",
)
