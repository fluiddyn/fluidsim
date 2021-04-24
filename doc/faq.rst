FAQ
===

Applications: Can I use FluidSim for
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Wall bounded / multiphase / reactive flows ...?

   Maybe. The built-in :mod:`solvers <fluidsim.solvers>` excels solving within
   periodic domains with pseudospectral solvers. However, FluidSim is a
   framework, and this allows FluidSim to interface with third-party solvers.
   See for instance :mod:`fluidsim.base.basilisk`, :mod:`fluidsim.base.dedalus`
   and `snek5000 <https://snek5000.readthedocs.io>`__.


Troubleshooting installation issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure you read the `installation guide <install>`__ carefully.

.. admonition:: Permission denied while running ``pip install fluidsim`` from
   PyPI or ``make develop`` inside the repository.

   This means you are probably using the Python bundled with the system and as
   a user you are restricted from installing packages into it. If this is so,
   create a `virtual environment`_.

.. _virtual environment: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment


.. admonition::  *No module named pip* or ``distutils.errors.DistutilsError``

   Package manager ``pip`` was not installed into your environment or is too
   old. The following commands should help::

      python -m ensurepip
      python -m pip install --upgrade pip setuptools wheel

.. admonition:: System freezes or becomes unresponsive as fluidsim starts to
   build extensions

   By default ``pythran`` extensions try to use ``gcc`` and this is a CPU and
   memory intensive compilation. Instead ``pythran`` can be configured to use
   ``clang``. See :ref:`pythranrc` for more details.

   Additionally to reduce the load during installation is to configure certain
   :ref:`build specific environment variables <env_vars>`.

.. admonition:: ``ModuleNotFoundError: No module named 'fluidsim_core. ...``

   FluidSim depends on :mod:`fluidsim_core` and both follow the same
   versioning. Make sure the versions match if you had used ``pip install`` or
   ``conda install``. For developers, ``make develop`` should install both as
   editable installations.
