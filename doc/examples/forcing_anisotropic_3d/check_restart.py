from fluiddyn.util import mpi

from fluidsim.util.scripts.turb_trandom_anisotropic import main
from fluidsim.util.scripts.restart import restart

nz = 24
t_end = 6
# params_a, sim_a = main(N=10, t_end=t_end, nz=nz)

params_b0, sim_b0 = main(N=10, t_end=3., nz=nz)
params_b1, sim_b1 = restart(args=[sim_b0.output.path_run], t_end=3.2)


mpi.printby0(
    f"""
# To visualize with fluidsim:

cd {sim_b1.output.path_run}
ipython --matplotlib -i -c "from fluidsim import load; sim = load()"
"""
)
