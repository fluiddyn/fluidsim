"""Standard output (:mod:`fluidsim.solvers.plate2d.output.print_stdout`)
==============================================================================


Provides:

.. autoclass:: PrintStdOutPlate2D
   :members:
   :private-members:

"""

from fluidsim.solvers.ns2d.output.print_stdout import PrintStdOutNS2D


class PrintStdOutPlate2D(PrintStdOutNS2D):
    """Used to print in both the stdout and the stdout.txt file, and also
    to print simple info on the current state of the simulation.

    """
