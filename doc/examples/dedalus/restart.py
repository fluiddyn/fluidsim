from pathlib import Path

import fluidsim as fls

paths = sorted((Path(fls.FLUIDSIM_PATH) / "examples").glob("dedalus_*"))
path = paths[-1]

params, Simul = fls.load_for_restart(path)

params.time_stepping.t_end += 10

sim = Simul(params)

sim.time_stepping.start()
