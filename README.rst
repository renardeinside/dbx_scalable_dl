Nocturne - utilities for Scalable Deep Learning with Spark, Horovod and Petastorm
=================================================================================

This repository contains useful elements and building blocks for scalable Deep Learning applications.

|build| |codecov| |black| |pypi|

.. |build| image:: https://img.shields.io/github/workflow/status/renardeinside/nocturne/CI%20pipeline/main?style=for-the-badge
    :alt: GitHub Workflow Status
    :target: https://github.com/renardeinside/dbx_scalable_dl/actions/workflows/onpush.yml

.. |pypi| image:: https://img.shields.io/pypi/v/nocturne.svg?style=for-the-badge
    :target: https://pypi.org/project/nocturne/
    :alt: Latest Python Release

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
    :target: https://github.com/psf/black
    :alt: We use black for formatting

.. |codecov| image:: https://img.shields.io/codecov/c/github/renardeinside/nocturne/main?style=for-the-badge&token=P9CiNFvruh
    :alt: Codecov branch
    :target: https://app.codecov.io/gh/renardeinside/nocturne


Dependencies
------------

The following libraries shall be installed before using :code:`nocturne`:

* JDK 1.8
* Apache Spark 3.x
* Tensorflow 2.x
* :code:`horovod[spark, tensorflow]`
* :code:`petastorm`

Since packaging of these dependencies might be challenging, you can use the base docker image with all dependencies provided in `Dockerfile.base`_.


To install the library, run:

.. code-block::

    pip install nocturne




Resources
---------

* `Horovod installation guide <https://horovod.readthedocs.io/en/stable/install_include.html>`_


.. _Dockerfile.base: docker/Dockerfile.base