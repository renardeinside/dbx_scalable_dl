Scalable Deep Learning on Databricks
====================================

This repository contains useful elements and building blocks for scalable Deep Learning applications on Databricks.

|build| |codecov| |black|

.. |build| image:: https://github.com/renardeinside/dbx_scalable_dl/actions/workflows/onpush.yml/badge.svg?branch=main
    :target: https://github.com/renardeinside/dbx_scalable_dl/actions/workflows/onpush.yml

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: We use black for formatting

.. |codecov| image:: https://codecov.io/gh/renardeinside/dbx_scalable_dl/branch/main/graph/badge.svg?token=P9CiNFvruh
    :target: https://codecov.io/gh/renardeinside/dbx_scalable_dl


Local environment setup
-----------------------


To easily setup local development environment, please use the `Dockerfile.dev`. 

For tests, please use:

.. code-block::

    make test

TBD
---

- add deployer job

Resources
---------

* `Horovod installation guide <https://horovod.readthedocs.io/en/stable/install_include.html>`_
* `MLflow custom Python Models <https://mlflow.org/docs/1.6.0/python_api/mlflow.pyfunc.html>`_
* `Amazon datasets <https://nijianmo.github.io/amazon/index.html>`_