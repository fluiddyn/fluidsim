#!/usr/bin/env python
#coding=utf8
# esubmit -n1 -t15 ./job_mpi_python simul_from_file_arg.py t_end=120.2 solver=SW1lwaves path_dir=/afs/pdc.kth.se/home/a/augier/Storage/Results_SW1lw/Pure_standing_waves_240x240/SE2D_SW1lwaves_forcingw_L=50.x50._240x240_c=700_f=0_2013-07-10_17-44-01/State_phys_480x480

import numpy as np
from solveq2d import solveq2d

import sys

def variable_from_args(strvar):
    for arg in sys.argv:
        if arg.startswith(strvar):
            var = arg.replace(strvar, '', 1)
            try:
                var = float(var)
            except ValueError:
                pass
    try: var
    except NameError:
        raise ValueError('One of the arg have to start with \"'+strvar+'\"')
    return var


t_end = variable_from_args("t_end=")
name_solver = variable_from_args("solver=")
path_dir = variable_from_args("path_dir=")


solver = solveq2d.import_module_solver_from_name(name_solver)

# choose the file with the time closer to t_approx
t_approx = 10.e10
name_file = solveq2d.name_file_from_time_approx(path_dir, t_approx)

path_file=path_dir+'/'+name_file

param = solveq2d.Param(path_file=path_file)

param['NEW_DIR_RESULTS']   = True
param['t_end']             = t_end

Lh = param['Lx']
k_f = 2*np.pi*param['nkmax_forcing']/Lh
P_Z = param['forcing_rate']
P_E = P_Z/k_f**2
delta_x = param['Lx']/param['nx']

# dissipation at the smallest scales:
# k^{-3}:    nu_n \simeq P_Z^{1/3} {\delta_x}^{n}
# k^{-5/3}:  nu_n \simeq P_E^{1/3} {\delta_x}^{n-2/3}
param['nu_8']              = 4.*10e-2*P_E**(1./3)*delta_x**(8-2./3)

# dissipation at the largest scales:
# param['nu_m4']             = 8.*10e0*P_E**(1./3)*(Lh/2)**(-4-2./3)

param['FORCING'] = True

param['period_print_simple']       = 0.001

param['period_spatial_means']      = 0.01
param['period_save_state_phys']    = 1.
param['period_save_spectra']       = 0.25
param['period_save_seb']           = 0.25
param['period_save_pdf']           = 0.5
param['period_save_incr']          = 0.25
param['SAVE_time_sigK']            = True

param['ONLINE_PLOT_OK']            = False
param['period_plot_field']         = 0.
param['field_to_plot']             = 'rot'
param['PLOT_spectra']              = False
param['PLOT_spatial_means']        = False
# param['PLOT_seb']                  = False
param['PLOT_incr']                  = True

param['type_flow_init']    ='LOAD_FILE'
param['dir_load']          = path_dir
param['file_load']         = name_file

sim = solver.Simul(param)

# sim.output.phys_fields.plot(numfig=0)
sim.time_stepping.start()
# sim.output.phys_fields.plot(numfig=1)

# solveq2d.show()
