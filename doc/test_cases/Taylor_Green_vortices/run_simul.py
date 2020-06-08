"""Taylor-Green Vortex at Re = 1600
===================================

https://www.grc.nasa.gov/hiocfd/wp-content/uploads/sites/22/case_c3.3.pdf


"""

import numpy as np

from fluidsim.solvers.ns3d.solver import Simul

params = Simul.create_default_params()

Re = 1600
V0 = 1.0
L = 1
params.nu_2 = V0 * L / Re

params.init_fields.type = "in_script"

params.time_stepping.t_end = 20.0 * L / V0

nx = 256 * 2
params.oper.nx = params.oper.ny = params.oper.nz = nx
lx = params.oper.Lx = params.oper.Ly = params.oper.Lz = 2 * np.pi * L

params.output.sub_directory = "taylor-green"

params.output.periods_print.print_stdout = 0.5

params.output.periods_save.phys_fields = 4
params.output.periods_save.spatial_means = 0.2
params.output.periods_save.spectra = 0.5
params.output.periods_save.spect_energy_budg = 0.5

params.output.spectra.kzkh_periodicity = 1


def init_simul(params):

    sim = Simul(params)

    X, Y, Z = sim.oper.get_XYZ_loc()

    vx = V0 * np.sin(X / L) * np.cos(Y / L) * np.cos(Z / L)
    vy = -V0 * np.cos(X / L) * np.sin(Y / L) * np.cos(Z / L)
    vz = sim.oper.create_arrayX(value=0)

    sim.state.init_statephys_from(vx=vx, vy=vy, vz=vz)

    sim.state.statespect_from_statephys()
    sim.state.statephys_from_statespect()

    return sim


def run_simul(sim):

    sim.time_stepping.start()

    from fluiddyn.util.mpi import printby0

    printby0(
        f"""
To visualize the output with Paraview, create a file states_phys.xmf with:

fluidsim-create-xml-description {sim.output.path_run}

# To visualize with fluidsim:

cd {sim.output.path_run}
ipython --matplotlib

# in ipython:

from fluidsim import load_sim_for_plot as load
sim = load()

sim.output.spatial_means.plot()
sim.output.spectra.plot1d(tmin=12, tmax=16, coef_compensate=5/3)

sim.output.phys_fields.set_equation_crosssection(f'x={{sim.oper.Lx/4}}')
sim.output.phys_fields.plot(field="vx", time=10)

sim.output.phys_fields.animate('vx')

"""
    )


if __name__ == "__main__":

    sim = init_simul(params)

    # only useful to plot fields before time_stepping
    # sim.output.init_with_initialized_state()

    run_simul(sim)
