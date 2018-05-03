Run a simple simulation and launch it on a cluster
==================================================

Of course online plotting slows down the simulation, so for larger simulation
we just remove the plot command from the script, which gives:

.. literalinclude:: simul_ns2d.py

This script can be launched by the commands ``python simul_ns2d.py``
(sequential) and ``mpirun -np 4 python simul_ns2d.py`` (parallel on 4
processes).

To submit the simulation on a cluster (here on one node):

.. literalinclude:: launch_simul.py
