Benchmarking and profiling dedalus solvers
==========================================

https://bitbucket.org/dedalus-project/dedalus

Progress
--------

 - Installation documented, see `here
   <https://dedalus-project.readthedocs.io/en/latest/installation.html#manual-installation>`_.

 - No built-in solvers, but examples are provided.

 - Solver initialization, and time-stepping is slow as grid size increases.

 - Benchmarking scripts to solve ns2d (primitive and stream-vorticity formulation),
   ns2dstrat (primitive and vorticity formulation) added.

.. todo::

        - Add ns3d solver.
        - Add profile script
        - Plot profiles, reusing code from fluidsim.util.console?
