Run a third party solver using FluidSim: Basilisk
=================================================

A proof of concept on how FluidSim can act as a framework to extend third party
solvers. Provided `Basilisk <http://basilisk.fr>`__ is installed and available for import
the script can be launched as ``python simul_third_party_basilisk.py`` from the
``doc/examples/basilisk`` directory.

.. literalinclude:: basilisk/simul_basilisk.py

The most advanced and robust package built on Fluidsim (core) is actually
`Snek5000 <https://github.com/exabl/snek5000/>`_, which is a framework to write
and use `NEK5000 <https://nek5000.mcs.anl.gov/>`__ solvers.
