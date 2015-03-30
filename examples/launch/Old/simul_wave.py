#!/usr/bin/env python
#coding=utf8
# 
# run simul_noise.py
# mpirun -np 8 python simul_noise.py

key_solver = 'NS2D'
key_solver = 'SW1l'
key_solver = 'SW1l.onlywaves'
key_solver = 'SW1l.exactlin'

import numpy as np
import fluiddyn as fld

solver = fld.import_module_solver_from_key(key_solver)
param = fld.Param()


nh = 64
param['nx'] = nh
param['ny'] = nh

param['f']                 = 5.
param['c2']                = (2.*np.pi)**2    # c = sqrt(g*H)

param['t_end']             = 0.4

ik_wave = 4
Lh = np.round(2*np.pi*ik_wave)

param['Lx'] = Lh
param['Ly'] = Lh

param['short_name_type_run']    = 'wave'

param['type_flow_init']    = 'WAVE'
param.ikx = ik_wave
param.eta0 = 0.01
param['deltat0']                   = 2.e-2
param['USE_CFL']                   = False



param['period_print_simple']       = 0.1
param['period_spatial_means']      = 0.01
param['period_save_state_phys']    = 10.
# param['period_save_spectra']       = 0.25
# param['period_save_SEB']           = 0.1
# param['period_save_pdf']           = 0.5

param['SAVE_time_sigK']            = True

# param['ONLINE_PLOT_OK']            = False
param['period_plot_field']         = 0.2
param['field_to_plot']             = 'eta'
param['PLOT_spatial_means']        = True
# param['PLOT_spectra']              = True
# param['PLOT_SEB']                  = True
# param['PLOT_pdf']                  = True


sim = solver.Simul(param)

# sim.output.phys_fields.plot(numfig=0, key_field='rot')
# sim.output.phys_fields.plot(numfig=1, key_field='h')



# print 'ap_fft', sim.state('ap_fft')[0, ik_wave]
# print 'am_fft', sim.state('am_fft')[0, ik_wave]

sim.time_stepping.start()

# print 'ap_fft', sim.state('ap_fft')[0, ik_wave]
# print 'am_fft', sim.state('am_fft')[0, ik_wave]



sim.output.phys_fields.plot(numfig=2, key_field='div')
sim.output.phys_fields.plot(numfig=3, key_field='eta')
sim.output.phys_fields.plot(numfig=4, key_field='h')
sim.output.phys_fields.plot(numfig=5, key_field='rot')



fld.show()

