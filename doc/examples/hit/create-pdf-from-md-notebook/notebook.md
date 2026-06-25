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

Here is a notebook containing the main output necessary to analyse a simulation.

# Description of a simulation

First, lets describe which simulation we analyse.

## Import module and loading paths

Here are the important modules for the following:

```{code-cell}
import os
import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

from fluidsim import load
```

Here we load the simulation with the corresponding simulation path:

```{code-cell} ipython3
simu_path = Path(os.environ.get("PATH_SIMUL_DIR", None))
print(f"{simu_path = }")
sim = load(simu_path, hide_stdout=True)
```


## Dimensionless number

In this section we list the important dimensionless numbers in two ways: from input and output values.

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

Select the min and max values depending on the simulation.

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

Here, we compute usefull quantities in the physical space.

### Velocity fields

Longitudinal velocity component $v_x$ on a horizontal cut at $z = 0$: 

```{code-cell} ipython3
fig, ax = sim.output.figure_axe()
sim.output.phys_fields.plot(QUIVER=False, numfig=fig.number, type_plot="pcolor",equation="z=0")
# filename = graph_path / f"phys_field_z=0_{N}_{nx}.png"
# fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
```

Longitudinal velocity component $v_x$ on a vertical cut at $y = 0$: 

```{code-cell} ipython3
fig, ax = sim.output.figure_axe()
sim.output.phys_fields.plot(equation="y=0", QUIVER=False, numfig=fig.number, type_plot="pcolor")
# filename = graph_path / f"phys_field_y=0_{N}_{nx}.png"
# fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
```

### Energy

Total energy in the domain and energy dissipation as functions of time: 

```{code-cell} ipython3
sim.output.spatial_means.plot()
fig_nums = plt.get_fignums()
fig_energy = plt.figure(fig_nums[-2])
fig_dissipation = plt.figure(fig_nums[-1])

# fig_energy.savefig(graph_path / f"energy_{N}_{nx}.pdf", dpi=300, bbox_inches='tight')
# fig_dissipation.savefig(graph_path / f"diss_{N}_{nx}.pdf", dpi=300, bbox_inches='tight')
```

## Spectral quantities

Here, we compute usefull quantities in the spectral space.

### Transfert and cumulated dissipation

Nonlinear turbulent energy transfer and cumulated energy spectra:  

```{code-cell} ipython3
fig_pi = sim.output.spect_energy_budg.plot_fluxes(tmin=tmin, tmax=tmax)
# fig_pi.savefig(graph_path / f"Pi_{N}_{nx}.pdf", dpi=300, bbox_inches='tight')
```
### Energy spectra

One dimensional energy spectra. If the simulation is stratified, one of the spectra is horizontal and the other is vertical.

```{code-cell} ipython3
if N is not None:
    directions="hz"
else:
    directions=None
fig_spectra = sim.output.spectra.plot1d(tmin=tmin, tmax=tmax, directions=directions, coef_compensate=5/3, coef_plot_k53=3, coef_plot_k3=300, ylim=(1e-2, 4))

# filename = graph_path / f"spectra_1d_{N}_{nx}.pdf"
# fig_spectra.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)


```

## Structure functions

Here, we compute usefull high order statistical quantities.

### Radial dependency

First, we longitudinal radial scalar function $\langle \mathbf{J}\cdot\mathbf{r}/r \rangle_{\theta,\phi}(r)$ normalized by $-\epsilon r$ to compare with the $4/3$-rd law. Note that in the case of a stratified fluid, this quantity has a kinetic and a potential components:

```{code-cell} ipython3
sim.output.kolmo_law.plot_radial_dependencies(tmin=tmin, tmax=tmax, which_plot='J')
```
### Cylindrical depency

Now, we take a look at $\nabla \cdot \mathbf{J} (r_h, r_v)$ normalized by $-4\epsilon$ which is what it is supposed to bet in the inertial range.

In log-log scale:
```{code-cell} ipython3
if N is not None:
    plotted='div_J'
else:
    plotted='div_JK'
print(f"{plotted=}")
sim.output.kolmo_law.plot_hv_dependencies(tmin=tmin, tmax=tmax, vmax=1, which_plot=plotted)
```

In linear scale:
```{code-cell} ipython3
sim.output.kolmo_law.plot_hv_dependencies(tmin=tmin, tmax=tmax, vmax=1, logscale=False, which_plot=plotted)
```

### Vectorial plots

Here, we directly take a look at the vectorial field $\mathbf{J} (r_h, r_v)$ normalized by $-4\epsilon$.
We first plot it almost on the full radial range: 


```{code-cell} ipython3
if N is not None:
    plotted='J'
    aniso_param=-0.1
else:
    plotted='JK'
    aniso_param=1
print(f"{plotted=}")
sim.output.kolmo_law.plot_Jhv_vector(tmin=tmin, tmax=tmax, which_plot=plotted, ani_param=aniso_param, logscale=False, ratio_vectors=10, shifted=False)
```

Then we zoom into the inertial range and plot in grey the vectorial field obtained with the following function:
$$
\mathbf{F}(\mathbf{r_h}, \mathbf{r_v}, \mathbf{\alpha}) = \frac{1}{\alpha + 2}(r_h\mathbf{e_h} + \alpha r_v\mathbf{e_v}),
$$

with $\alpha$ an anisotropic parameter. Note that in the plot the amplitudes of the "theretical vectors" are normalized by the amplitudes of the real vectors
so that only direction is compared. 


```{code-cell} ipython3
sim.output.kolmo_law.plot_Jhv_vector(tmin=tmin, tmax=tmax, which_plot="J", theory=True, vect_theory=True, ani_param=aniso_param, logscale=False, ratio_vectors=10, shifted=False)
```
