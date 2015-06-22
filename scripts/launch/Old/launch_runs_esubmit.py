"""Launch many runs from a base directory containing."""

import numpy as np
from solveq2d import solveq2d
import sys
import os

nh_approach = 3840
nh = nh_approach*2
t_end = 196.

# -n1Q for 1 Q node (64 GB Ram)
# command_base = "esubmit -n1Q -t10 ./job_mpi_python simul_from_file_arg.py"
command_base = "esubmit -n1Q -t11520 ./job_mpi_python simul_from_file_arg.py"
# command_base = "python simul_from_file_arg.py"

command_base = (
    command_base
    +' t_end='+repr(t_end)
    )

dir_base = (
        '~/Storage/Results_SW1lw/'
        'Pure_standing_waves_'+repr(nh_approach)+'x'+repr(nh_approach)
)

set_of_dir_results = solveq2d.SetOfDirResults(dir_base=dir_base)

values_c2 = set_of_dir_results.values_c2
values_c = np.sqrt(values_c2)
values_c = np.array([40.])
# values_c = np.array([10, 20, 40, 70, 100, 200, 400, 700, 1000])

values_solver = set_of_dir_results.values_solver
values_solver = ['SW1lwaves']

values_c2 = values_c**2
tuple_loop =  [(c2,name_solver) 
               for c2 in values_c2 
               for name_solver in values_solver]
for c2, name_solver in tuple_loop:
    path_dir = set_of_dir_results.one_path_from_values(
        solver=name_solver,
        c2=c2)

    command = (
        command_base+
        ' solver='+name_solver+
        ' path_dir='+path_dir+'/State_phys_'+repr(nh)+'x'+repr(nh)
        )

    print "run command:\n", command
    os.system(command)
