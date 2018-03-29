#!/usr/bin/env python
#coding=utf8
# 
# nohup mpirun -np 4 python -u loop_simuls_from_file.py &
# nohup mpirun -np 8 python -u loop_simuls_from_file.py &

import numpy as np
from solveq2d import solveq2d

def modify_param(param):
    """Modify the param object for the new simul."""
    param['NEW_DIR_RESULTS']   = True
    param['t_end']             = 140.

    Lh = param['Lx']
    k_f = 2*np.pi*param['nkmax_forcing']/Lh
    P_Z = param['forcing_rate']
    P_E = P_Z/k_f**2
    delta_x = param['Lx']/param['nx']

    # param['FORCING'] = True

    # dissipation at the smallest scales:
    # k^{-3}:    nu_n \simeq P_Z^{1/3} {\delta_x}^{n}
    # k^{-5/3}:  nu_n \simeq P_E^{1/3} {\delta_x}^{n-2/3}
    param['nu_8']              = 4.*10e-2*P_E**(1./3)*delta_x**(8-2./3)

    # dissipation at the largest scales:
    param['nu_m4']             = 8.*10e0*P_E**(1./3)*(Lh/2)**(-4-2./3)

    param['period_print_simple']       = 0.05
    param['period_spatial_means']      = 0.01
    param['period_save_state_phys']    = 2.
    param['period_save_spectra']       = 0.25
    param['period_save_seb']           = 0.25
    param['period_save_pdf']           = 0.5
    param['period_save_incr']          = 0.25
    param['SAVE_time_sigK']            = True

    param['ONLINE_PLOT_OK']            = False

    param['type_flow_init']    ='LOAD_FILE'
    param['dir_load']          = path_dir
    param['file_load']         = name_file


dict_solvers = solveq2d.ModulesSolvers(['SW1lexlin'])

t_approx = 10.e10
nh = 1024

nh_approach = nh/4
# dir_base = 'Approach_runs_'+repr(nh_approach)+'x'+repr(nh_approach)
dir_base = 'Waves_standing_'+repr(nh_approach)+'x'+repr(nh_approach)

# dir_base = '/scratch/augier/Results_for_article_SW1l/'+dir_base
# print dir_base

set_of_dir_results = solveq2d.SetOfDirResults(dir_base=dir_base)

values_c2 = set_of_dir_results.values_c2
# values_c2 = [400.]
values_solver = set_of_dir_results.values_solver
values_solver = ['SW1lexlin']

tuple_loop =  [(c2,name_solver) 
               for c2 in values_c2 
               for name_solver in values_solver]
for c2, name_solver in tuple_loop:
    path_dir = set_of_dir_results.one_path_from_values(
        solver=name_solver,
        c2=c2)
    path_dir = path_dir+'/State_phys_'+repr(nh)+'x'+repr(nh)

    name_file = solveq2d.name_file_from_time_approx(path_dir, t_approx)
    path_file=path_dir+'/'+name_file
    param = solveq2d.Param(path_file=path_file)
    modify_param(param)

    solver = dict_solvers[name_solver]
    sim = solver.Simul(param)
    sim.time_stepping.start()
    del(sim)

