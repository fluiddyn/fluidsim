"""
init_fields_energyK_0.py
=========================

# Environment
-------------
Python 3.6 (Conda)

# Description:
--------------
Energy = 0 in init_fields in MPI. 

It works well in SEQUENTIAL.
It does not work in MPI: neither cluster neither pc.
It does not work for init_fields = 'dipole'; init_fields = 'noise'
It does not work for ns2d nor ns2d.strat.

# Result expected
-----------------
Energy != 0 in it=0 in MPI.

# Traceback: 
-----------
Beginning of the computation
    compute until it =        1
it =      0 ; t =          0 ; deltat  =        0.2
              energy = 0.000e+00 ; Delta energy = +0.000e+00
<fluidsim.operators.operators2d.OperatorsPseudoSpectral2D object at 0x7fc8bda5f908>

# Identification
----------------
It seems to be fft2() method of the class
fluidsim.operators.operators2d.OperatorsPseudoSpectral2D

After using fft2(), the arrays look empty!!!

# To run the bug:
----------------

mpirun -np 2 python init_fields_energyK_0.py
 
"""
from __future__ import print_function

from fluidsim.solvers.ns2d.solver import Simul

def _create_object_params():
    params = Simul.create_default_params()

    # Operator parameters
    params.oper.nx = params.oper.ny = 128

    # Time stepping parameters
    params.time_stepping.USE_CFL = False
    params.time_stepping.USE_T_END = False
    params.time_stepping.it_end = 1
    
    # Output parameters
    params.output.HAS_TO_SAVE = False
    params.output.periods_print.print_stdout = 1e-15
    return params

init_fields = 'noise'
print('***** init_fields {} *******'.format(init_fields))
params = _create_object_params()
params.init_fields.type = init_fields
sim = Simul(params)
sim.time_stepping.start()


