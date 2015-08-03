#!/usr/bin/env python
#coding=utf8
# 
# run simul_forcing.py
# mpirun -np 8 python simul_forcing.py
# nohup mpirun -np 4 python -u simul_forcing.py

key_solver = 'NS2D'
key_solver = 'SW1l'
key_solver = 'SW1l.onlywaves'
key_solver = 'SW1l.exactlin'

import numpy as np
import fluiddyn as fld

solver = fld.import_module_solver_from_key(key_solver)
param = fld.Param()

param.short_name_type_run = 'forcing'


nh = 2*64
param['nx'] = nh
param['ny'] = nh

param['f']                 = 0.
param['c2']                = 200.    # c = sqrt(g*H)

param['t_end']             = 1.


param['FORCING']           = False
param['type_forcing']      = 'WAVES'
param['nkmax_forcing']     = 8
param['nkmin_forcing']     = 5
param['forcing_rate']      = 1.

Lh = np.round(2*np.pi*param['nkmax_forcing'])
param['Lx'] = Lh
param['Ly'] = Lh

k_f = 2*np.pi*param['nkmax_forcing']/Lh
P_Z = param['forcing_rate']
P_E = P_Z/k_f**2
delta_x = param['Lx']/param['nx']

# dissipation at the smallest scales:
# k^{-3}:    nu_n \simeq P_Z^{1/3} {\delta_x}^{n}
# k^{-5/3}:  nu_n \simeq P_E^{1/3} {\delta_x}^{n-2/3}
param['nu_8']              = 2.*10e-2*P_E**(1./3)*delta_x**(8-2./3)

# dissipation at the largest scales:
param['nu_m4']             = 8.*10e0*P_E**(1./3)*(Lh/2)**(-4-2./3)

param['coef_dealiasing']   = 8./9

param['type_flow_init']    = 'NOISE'
param['lambda_noise']      = 2*np.pi/k_f
param['max_velo_noise']    = 0.000001


param['period_print_simple']       = 2.
param['period_spatial_means']      = 0.01
param['period_save_state_phys']    = 10.
param['period_save_spectra']       = 0.5
param['period_save_seb']           = 0.1
param['period_save_pdf']           = 0.1
param['period_save_incr']          = 0.1
param['SAVE_time_sigK']            = True

# param['ONLINE_PLOT_OK']            = False
param['period_plot_field']         = 0.
param['field_to_plot']             = 'eta'
# param['PLOT_spatial_means']        = True
param['PLOT_spectra']              = True
param['PLOT_seb']                  = True
param['PLOT_pdf']                  = True
param['PLOT_incr']                  = True

# import resource

# np.seterr(all='raise')

sim = solver.Simul(param)

# sim.output.phys_fields.plot(numfig=0)
sim.time_stepping.start()
# sim.output.phys_fields.plot(numfig=1)


fld.show()

