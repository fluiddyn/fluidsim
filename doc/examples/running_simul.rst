Running a simple simulation
===========================

This script is an example of how one can run a small simulation with
instantaneous plotting. This can be useful for simple tests. It can be
launched by the commands ``python simul_ns2d.py`` (sequencial) and
``mpirun -np 8 python simul_ns2d.py`` (parallel on 8 processes).

.. literalinclude:: simul_ns2d_plot.py

Of course the plots slow down the simulation, so for larger simulation
we just remove the plot command from the script, which gives:

.. literalinclude:: simul_ns2d.py

To submit the simulation on a cluster (here on one node), just run
this tiny script:

.. literalinclude:: launch_simul.py
