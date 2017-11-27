Oneliner commands
=================

The commands listed below were used to generate the benchmarks and profiles
(to be) reported in the article on FluidSim. It should work on any workstation
or any configured SLURM job-scheduler based cluster. Add the flag ``-n`` or
``--dry-run`` flag to see what the commands look like without executing them.

Note: for profiles, simply replace ``submit_strong`` with ``submit_profile``::

ns2d benchmarks (strong) / profiles
-----------------------------------

.. code-block:: bash

   ./submit_strong -d 2 1024 1024 -xn 8

ns3d benchmarks (strong) / profiles
------------------------------------

.. code-block:: bash

   ./submit_strong -d 3 128 128
   ./submit_strong -d 3 256 256 -nc 4 -m inter-intra
   ./submit_strong -d 3 512 512 -nn 2 -xn 16 -m inter
   ./submit_strong -d 3 1024 1024 -nn 16 -xn 32 -m inter
