"""
no_attribute_has_uxuy.py
========================

# Environment:
-------------
Python 3.6

# Description of the bug:
-------------------------
PhysFieldsBase2D object has no attribute to _has_uxuy.

# Traceback:
------------
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
~/Dev/fluidsim/scripts/bugs/no_attribute_has_uxuy.py in <module>()
     43 sim.time_stepping.start()
     44 
---> 45 sim.output.phys_fields.plot(time=1.5)

~/Dev/fluidsim/fluidsim/base/output/phys_fields.py in plot(self, field, key_field, time, QUIVER, vecx, vecy, nb_contours, type_plot, iz, vmin, vmax, cmap, numfig)
    639 
    640             if QUIVER:
--> 641                 field, vecx, vecy = self._ani_get_field(time, key_field)
    642             else:
    643                 field = self._ani_get_field(time, key_field, need_uxuy=False)

~/Dev/fluidsim/fluidsim/base/output/phys_fields.py in _ani_get_field(self, time, key, need_uxuy)
    383             field = f['state_phys'][key].value
    384 
--> 385             if need_uxuy and self._has_uxuy:
    386                 try:
    387                     ux = f['state_phys']['ux'].value

AttributeError: 'PhysFieldsBase2D' object has no attribute '_has_uxuy'

# Steps to reproduce the bug:
-----------------------------
Run in a terminal the following script:

"""

from __future__ import print_function

from fluidsim.solvers.ns2d.strat.solver import Simul

params = Simul.create_default_params()
params.oper.nx = params.oper.ny = 32

params.init_fields.type = 'noise'

# Time stepping parameters
params.time_stepping.USE_CFL = True
params.time_stepping.t_end = 2.

# Output parameters
params.output.HAS_TO_SAVE = True
params.output.periods_save.phys_fields = 0.2

sim = Simul(params)
sim.time_stepping.start()

sim.output.phys_fields.plot(time=1.5)
