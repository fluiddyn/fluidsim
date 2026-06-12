---
authors: ["Clovis Lambert", "Pierre Augier"]
abstract: |
  A executable notebook to analyse a simulation
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
exports:
  - format: typst
    template: lapreprint-typst
execute:
  depends_on_env: ["PATH_SIMUL_DIR"]

---

# Description of a simulation

```{code-cell}
import os
import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

from fluidsim import load

from util import get_spectra_values_kh, get_spectra_values_kz, load_temp_average
```

```{code-cell} python
path_simul_dir = os.environ.get("PATH_SIMUL_DIR", None)
print(f"{path_simul_dir = }")
```


## Import module and loading paths

```{code-cell} ipython3
simu_path = Path(os.environ.get("PATH_SIMUL_DIR", ""))
path_file_kolmo = simu_path / "kolmo_law.h5"
sim = load(simu_path, hide_stdout=True)
tmin = 5
%matplotlib ipympl
```


## Dimensionless number

### Quantities calculated from input values

```{code-cell} ipython3
from math import pi

try:
  N = sim.params.N
except AttributeError:
  N = None

nz = sim.params.oper.nz
Lx = sim.params.oper.Lx
Lz = sim.params.oper.Lz
delta_kz = 2 * np.pi / Lz

k_max = delta_kz * nz / 2
nu = sim.params.nu_2
eta_input = nu ** (3/4)
kmaxeta = k_max * eta_input

Re = 1 / nu

strat = False
if N is not None:
  Fh = 1 / (N) # injection_rate**1/3 / (Lfh**2/3 * N)
  Rb = Re * (Fh**2)
  strat = True
else:
  Fh = None
  Rb = None

print(f"{kmaxeta=}")
print(f"{Fh=}")
print(f"R{Re=}")
print(f"{Rb=}")
print(f"{nu=}")
```

```{code-cell} ipython3
if N is None:
  tmin = 66.6
  tmax = None 
else:
  tmin = 11.5
  tmax = None

print(f"{tmin=}")
print(f"{tmax=}")
```

### Quantities calculated from output values

```{code-cell} ipython3
sim.output.spatial_means.plot_dimless_numbers_versus_time()
```

```{code-cell} ipython3
dimless_numbers = sim.output.spatial_means.get_dimless_numbers_averaged(tmin=tmin, tmax=tmax)
dimless_numbers
```

```{code-cell} ipython3
sim.output.get_mean_values(tmin=tmin, tmax=tmax)
```

## Physical quantities

### Velocity fields

```{code-cell} ipython3
fig, ax = sim.output.figure_axe()
sim.output.phys_fields.plot(QUIVER=False, numfig=fig.number, type_plot="pcolor",equation="z=0")
# filename = graph_path / f"phys_field_z=0_{N}_{nx}.png"
# fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
```

```{code-cell} ipython3
fig, ax = sim.output.figure_axe()
sim.output.phys_fields.plot(equation="y=0", QUIVER=False, numfig=fig.number, type_plot="pcolor")
# filename = graph_path / f"phys_field_y=0_{N}_{nx}.png"
# fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
```

### Energy

```{code-cell} ipython3
sim.output.spatial_means.plot()
fig_nums = plt.get_fignums()
fig_energy = plt.figure(fig_nums[-2])
fig_dissipation = plt.figure(fig_nums[-1])

# fig_energy.savefig(graph_path / f"energy_{N}_{nx}.pdf", dpi=300, bbox_inches='tight')
# fig_dissipation.savefig(graph_path / f"diss_{N}_{nx}.pdf", dpi=300, bbox_inches='tight')
```

## Spectral quantities

### Transfert and cumulated dissipation

```{code-cell} ipython3
fig_pi = sim.output.spect_energy_budg.plot_fluxes(tmin=tmin, tmax=tmax)
# fig_pi.savefig(graph_path / f"Pi_{N}_{nx}.pdf", dpi=300, bbox_inches='tight')
```
### Energy spectra

```{code-cell} ipython3
if N is not None:
    directions="hz"
else:
    directions=None
fig_spectra = sim.output.spectra.plot1d(tmin=tmin, tmax=tmax, directions=directions, coef_compensate=5/3, coef_plot_k53=10, coef_plot_k3=10**3)

# filename = graph_path / f"spectra_1d_{N}_{nx}.pdf"
# fig_spectra.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)


```

## Structure functions

### Radial dependency

```{code-cell} ipython3
sim.output.kolmo_law.plot_radial_dependencies(tmin=tmin, tmax=tmax, which_plot='J')
```
### Cylindrical depency

```{code-cell} ipython3
sim.output.kolmo_law.plot_hv_dependencies(tmin=tmin, tmax=tmax, which_plot='div_J')
```

### Vectorial plots

```{code-cell} ipython3
sim.output.kolmo_law.plot_Jhv_vector(tmin=tmin, tmax=tmax, which_plot="J", logscale=False, ratio_vectors=10, shifted=False)
```

