Scalable Deep Learning on Databricks
====================================

This repository contains useful elements and building blocks for scalable Deep Learning applications on Databricks.

|build| |codecov| |black|

.. |build| image:: https://img.shields.io/github/workflow/status/renardeinside/dbx_scalable_dl/CI%20pipeline/main?style=for-the-badge
    :alt: GitHub Workflow Status
    :target: https://github.com/renardeinside/dbx_scalable_dl/actions/workflows/onpush.yml


.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
    :target: https://github.com/psf/black
    :alt: We use black for formatting

.. |codecov| image:: https://img.shields.io/codecov/c/github/renardeinside/dbx_scalable_dl/main?style=for-the-badge&token=P9CiNFvruh
    :alt: Codecov branch


Local environment setup
-----------------------


To easily setup local development environment, please use the `Dockerfile.dev`. 

For local tests, please use:

.. code-block::

    make local-test

TBD
---

- add metric collection:
    - implement a daemon-like process to collect information from Gangilia and export it
- add deployer job


Resources
---------

* `Horovod installation guide <https://horovod.readthedocs.io/en/stable/install_include.html>`_
* `MLflow custom Python Models <https://mlflow.org/docs/1.6.0/python_api/mlflow.pyfunc.html>`_
* `Amazon datasets <https://nijianmo.github.io/amazon/index.html>`_