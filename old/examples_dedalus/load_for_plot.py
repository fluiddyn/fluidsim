from pathlib import Path

import matplotlib.pyplot as plt

import fluidsim as fls

paths = sorted((Path(fls.FLUIDSIM_PATH) / "examples").glob("dedalus_*"))
path = paths[-1]

sim = fls.load_sim_for_plot(path)

sim.output.phys_fields.plot("b")

plt.show()
