from setuptools import find_packages, setup
from dbx_scalable_dl import __version__

# here we only mention the packages that are not available in Databricks ML Runtime
INSTALL_REQUIRES = ["tensorflow-recommenders==0.6.0", "dunamai"]

# setup_requires is set externally via pyproject.toml

setup(
    name="dbx_scalable_dl",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=INSTALL_REQUIRES,
    version=__version__,
    description="",
    author="",
)
