#!/usr/bin/env python
#coding=utf8
# 
# mpirun -np 8 python simul_dipole.py

key_solver = 'NS2D'
key_solver = 'SW1l'
key_solver = 'SW1l.onlywaves'
key_solver = 'SW1l.exactlin'

import numpy as np
import fluiddyn as fld

solver = fld.import_module_solver_from_key(key_solver)
param = fld.Param()

nh = 2*128
param['nx'] = nh
param['ny'] = nh
Lh = 10.
param['Lx'] = Lh
param['Ly'] = Lh

delta_x = param.Lx/param.nx
param['nu_8'] = 2.*10e-1*param.forcing_rate**(1./3)*delta_x**8
param['f']                         = 0.
param['coef_dealiasing']           = 2./3

param['t_end']                     = 1.

param['type_flow_init']            = 'DIPOLE'
param['short_name_type_run']       = 'dipole'

param['period_print_simple']       = 0.25
param['period_spatial_means']      = .02
param['period_save_state_phys']    = 0
param['period_save_spectra']       = 0.1
# param['period_save_seb']           = 0.1

# param['ONLINE_PLOT_OK']            = False
param['period_plot_field']         = 2.
param['field_to_plot']             = 'rot'
param['PLOT_spatial_means']        = True
param['PLOT_spectra']              = True
# param['PLOT_seb']                  = True

sim = solver.Simul(param)

sim.output.phys_fields.plot(numfig=0)
sim.time_stepping.start()
sim.output.phys_fields.plot(numfig=1)

fld.show()
