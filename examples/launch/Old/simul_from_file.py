#!/usr/bin/env python
#coding=utf8
# 
# run simul_from_file.py
# mpirun -np 8 python simul_from_file.py
# nohup mpirun -np 4 python -u simul_from_file.py &

key_solver = 'NS2D'
key_solver = 'SW1l'
key_solver = 'SW1l.onlywaves'
key_solver = 'SW1l.exactlin'

import numpy as np
import fluiddyn as fld

from fluiddyn.simul.util import name_file_from_time_approx


solver = fld.import_module_solver_from_key(key_solver)


#name_dir = ''
name_dir = (
'/afs/pdc.kth.se/home/a/augier/Storage'
'/Results_SW1lw'
'/Pure_standing_waves_1920x1920'
'/SE2D_SW1lwaves_forcingw_L=50.x50._1920x1920_c=20_f=0_2013-07-15_10-40-51'
)
# path_dir = solveq2d.path_dir_results+'/'+name_dir
path_dir = name_dir


# choose the file with the time closer to t_approx
t_approx = 10.e10
name_file = name_file_from_time_approx(path_dir, t_approx)
# or ...
#name_file = ''

path_file=path_dir+'/'+name_file

param = fld.Param(path_file=path_file)

param['NEW_DIR_RESULTS']   = True
param['t_end']             = 155.

Lh = param['Lx']
k_f = 2*np.pi*param['nkmax_forcing']/Lh
P_Z = param['forcing_rate']
P_E = P_Z/k_f**2
delta_x = param['Lx']/param['nx']

# dissipation at the smallest scales:
# k^{-3}:    nu_n \simeq P_Z^{1/3} {\delta_x}^{n}
# k^{-5/3}:  nu_n \simeq P_E^{1/3} {\delta_x}^{n-2/3}
param['nu_8']              = 4.*10e-2*P_E**(1./3)*delta_x**(8-2./3)
# param['nu_2']              = 1.5*10e-0*P_E**(1./3)*delta_x**(2-2./3)

# dissipation at the largest scales:
# param['nu_m4']             = 0. #8.*10e0*P_E**(1./3)*(Lh/2)**(-4-2./3)

param['FORCING'] = True



ROTATION = True
if ROTATION:
    Bu = 2.
    c = np.sqrt(param['c2'])
    Kf = 6*2*np.pi/Lh
    f = c*Kf/np.sqrt(Bu)
    param['f']                 = f


param['period_print_simple']       = 0.001
param['period_spatial_means']      = 0.01
param['period_save_state_phys']    = 2.
param['period_save_spectra']       = 0.2
param['period_save_seb']           = 0.2
param['period_save_pdf']           = 0.5
param['period_save_incr']          = 0.1
param['SAVE_time_sigK']            = False


param['ONLINE_PLOT_OK']            = False
# param['period_plot_field']         = 0.
# param['field_to_plot']             = 'rot'
# param['PLOT_spectra']              = True
# param['PLOT_spatial_means']        = False
# param['PLOT_seb']                  = False
# param['PLOT_incr']                 = False


param['type_flow_init']    ='LOAD_FILE'
param['dir_load']          = path_dir
param['file_load']         = name_file

sim = solver.Simul(param)

# sim.output.phys_fields.plot(numfig=0)
sim.time_stepping.start()
# sim.output.phys_fields.plot(numfig=1)

fld.show()
