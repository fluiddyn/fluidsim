#!/usr/bin/env python
#coding=utf8
# 
# nohup mpirun -np 8 python -u loop_simuls_forcing.py &

import numpy as np
from solveq2d import solveq2d

param = solveq2d.Param()

param.short_name_type_run = 'forcing'

nh = 2*128
param['nx'] = nh
param['ny'] = nh

param['f']                 = 0.

param['t_end']             = 140.


param['FORCING']           = True
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
param['nu_8']              = 3.*10e-2*P_E**(1./3)*delta_x**(8-2./3)

# dissipation at the largest scales:
param['nu_m4']             = 8.*10e0*P_E**(1./3)*(Lh/2)**(-4-2./3)




param['coef_dealiasing']   = 8./9

param['type_flow_init']    = 'NOISE'
param['lambda_noise']      = 2*np.pi/k_f
param['max_velo_noise']    = 0.01


param['period_print_simple']       = 2.
param['period_spatial_means']      = 0.1
param['period_save_state_phys']    = 20.
param['period_save_spectra']       = 0.5
param['period_save_seb']           = 0.5
param['period_save_pdf']           = 0.5
param['SAVE_time_sigK']            = True

param['ONLINE_PLOT_OK']            = False
# param['period_plot_field']         = 0.
# param['field_to_plot']             = 'rot'
# param['PLOT_spatial_means']        = False
# param['PLOT_spectra']              = True
# param['PLOT_seb']                  = True
# param['PLOT_pdf']                  = True


values_solver = ['SW1lexlin', 'SW1l']

dict_solvers = solveq2d.ModulesSolvers(values_solver)

values_c2 = [100, 200, 400, 600]


tuple_loop =  [(c2, name_solver) 
               for c2 in values_c2 
               for name_solver in values_solver]
for c2, name_solver in tuple_loop:

    param['c2'] = c2

    solver = dict_solvers[name_solver]

    sim = solver.Simul(param)
    sim.time_stepping.start()





# solver.plt.show(block=True)
# solver.plt.show()



