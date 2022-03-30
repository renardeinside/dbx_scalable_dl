Nocturne - utilities for Scalable Deep Learning with Spark, Horovod and Petastorm
=================================================================================

This repository contains useful elements and building blocks for scalable Deep Learning applications.

|build| |codecov| |black|

.. |build| image:: https://img.shields.io/github/workflow/status/renardeinside/nocturne/CI%20pipeline/main?style=for-the-badge
    :alt: GitHub Workflow Status
    :target: https://github.com/renardeinside/dbx_scalable_dl/actions/workflows/onpush.yml


.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
    :target: https://github.com/psf/black
    :alt: We use black for formatting

.. |codecov| image:: https://img.shields.io/codecov/c/github/renardeinside/dbx_scalable_dl/main?style=for-the-badge&token=P9CiNFvruh
    :alt: Codecov branch
    :target: https://app.codecov.io/gh/renardeinside/dbx_scalable_dl


Dependencies
------------

The following libraries shall be installed before using :code:`nocturne`:

* Spark 3.x
* Tensorflow 2.x
* :code:`horovod[spark, tensorflow]`
* :code:`petastorm`

Local environment setup
-----------------------


To easily setup local development environment, please use the `Dockerfile.dev`. 

For local tests, please use:

.. code-block::

    make local-test



Resources
---------

* `Horovod installation guide <https://horovod.readthedocs.io/en/stable/install_include.html>`_
* `MLflow custom Python Models <https://mlflow.org/docs/1.6.0/python_api/mlflow.pyfunc.html>`_
* `Amazon datasets <https://nijianmo.github.io/amazon/index.html>`_