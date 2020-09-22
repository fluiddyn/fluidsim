from fluidsim import load_for_restart, path_dir_results

path_dir_root = path_dir_results / "examples"

path_dir = sorted(path_dir_root.glob("NS2D_32x32_S2pix2pi*"))[-1]

params, Simul = load_for_restart(path_dir)

params.time_stepping.t_end += 2

sim = Simul(params)

sim.time_stepping.start()
