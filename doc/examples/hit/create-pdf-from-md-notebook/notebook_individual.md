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
  if abs(kmaxeta - 1.0) <= 0.1:
      tmin = 21.0
      tmax = 25.0
  elif kmaxeta < 0.5:
      tmin = 10.0
      tmax = 20.0
else:
  tmin = 6.5
  tmax = 8.0
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

for fig in [fig_energy, fig_dissipation]:
    for ax in fig.axes:
        ax.set_xlabel(f"{ax.get_xlabel()} $t$", fontsize=18, labelpad=10)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(20)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(bottom=0.2)

# fig_energy.savefig(graph_path / f"energy_{N}_{nx}.pdf", dpi=300, bbox_inches='tight')
# fig_dissipation.savefig(graph_path / f"diss_{N}_{nx}.pdf", dpi=300, bbox_inches='tight')
```

## Spectral quantities

```{code-cell} ipython3
sim.output.spect_energy_budg.plot_fluxes(tmin=tmin, tmax=tmax)
```

```{code-cell} ipython3
from fluidsim_core.output.base import SimReprMakerCore
name_run, summary = sim.output._sim_repr_maker.make_representations()
data = sim.output.spectra.load1d_mean(tmin=tmin, tmax=tmax)

spectra_kh = get_spectra_values_kh(data, strat)
spectra_kz = get_spectra_values_kz(data, strat)

coef_compensate = 5 / 3
kh = spectra_kh["kh"]
kz = spectra_kz["kz"]

fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(kh, spectra_kh["Ekh_K"] * kh**coef_compensate,
        label="$E_{K}(k_h)$", color="red")
ax.loglog(kz, spectra_kz["Ekz_K"] * kz**coef_compensate,
        label="$E_{K}(k_z)$", color="red", linestyle='--')
if strat:
  ax.loglog(kh, spectra_kh["Ekh_A"] * kh**coef_compensate,
          label="$E_{A}(k_h)$", color="blue")
  ax.loglog(kz, spectra_kz["Ekz_A"] * kz**coef_compensate,
          label="$E_{A}(k_z)$", color="blue", linestyle='--')


ax.set_xlabel("$k_h, k_z$", fontsize=24)
ax.set_ylabel(r"$E(k) \cdot k^{5/3}$", fontsize=24)
ax.set_title(f"spectra\n{summary}", fontsize=14)
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
ax.set_ylim(0.07, 4)


xlims = ax.get_xlim()
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals_53 = np.ones_like(x_vals)
y_vals_3 = 200 * x_vals**-3 * x_vals**coef_compensate

ax.loglog(x_vals, y_vals_53, 'k--', label=r'$\propto k^{-5/3}$')
ax.loglog(x_vals, y_vals_3, 'k-.', label=r'$\propto k^{-3}$')

legend = ax.legend()
for text in legend.get_texts():
    text.set_fontsize(20)

plt.tight_layout()
# filename = graph_path / f"spectra_kin_pot_kh_kz_{N}_{nx}.pdf"
# plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)


```

## Structure functions

Keys to plot in the following and average over time:

```{code-cell} ipython3
params = sim.params
keys_state_phys = sim.state.keys_state_phys

keys = [
    "S2_k_r",
    "divJ_k_r",
    "Jl_k_r",
    "Jl_k_hv", 
    "divJ_k_hv",
    "Jh_k_hv",
    "Jv_k_hv"
]


if "b" in keys_state_phys:
    keys.extend([
      "S2_p_r",
      "divJ_p_r",
      "Jl_p_r",
      "Jl_p_hv",
      "divJ_p_hv",
      "Jh_p_hv",
      "Jv_p_hv"
    ])

to_plot, tmin, tmax = load_temp_average(keys, tmin, tmax, path_file_kolmo)

title = f"$n_x={params.oper.nx}$"
```

Get $eta$ to normalize $r$, $E_K$ as asymptote for $S_2$ and $r$, $rh$ and $rv$:

```{code-cell} ipython3
dimless_num = dimless_numbers["dimensional"]
eta = dimless_num["eta"]
EK = dimless_num["EKh"] + dimless_num["EKz"]

with h5py.File(path_file_kolmo, "r") as file:
    r_store = np.array(file["r_store"])
    rh_store = np.array(file["rh_store"])
    rv_store = np.array(file["rv_store"])

RH, RV = np.meshgrid(rh_store, rv_store)
```

Get quantities that depends on $r$:

```{code-cell} ipython3
# Compensated plots
coef_comp3=1,
coef_comp2=2 / 3
Jl_k_comp = -to_plot["Jl_k_r"] / (r_store**coef_comp3)
divJ_k = to_plot["divJ_k_r"]
S2_k_comp = to_plot["S2_k_r"] / (r_store**coef_comp2)

# Theoretical values
Jl_k_th = 4 / 3 * np.ones_like(r_store)
S2_k_th = 22 / 3 * np.ones_like(r_store)
EK_array = EK * np.ones_like(r_store)

if "b" in keys_state_phys:
    Jl_p_comp = -to_plot["Jl_p_r"] / (r_store**coef_comp3)
    divJ_p = to_plot["divJ_p_r"]
    S2_p_comp = to_plot["S2_p_r"] / (r_store**coef_comp2)
```


Get quantities that depends on $rh$ and $rv$:

```{code-cell} ipython3
# Compute radius for normalization
radius = np.sqrt(RH**2 + RV**2)

Jk_l_comp = -to_plot["Jl_k_hv"] / (radius + 1e-14)
divJk_hv = to_plot["divJ_k_hv"]

if "b" in keys_state_phys:
    Jp_l_comp = -to_plot["Jl_p_hv"] / (radius + 1e-14)
    divJp_hv = to_plot["divJ_p_hv"]
```

Get vectorial quantities:

```{code-cell} ipython3
Jk_v = to_plot["Jv_k_hv"]
Jk_h = to_plot["Jh_k_hv"]
```


### S2(r)

Plot $S_2$(r):

```{code-cell} ipython3
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.set_ylabel(r"$S_2^K(r)/(r^{2/3}\epsilon^{2/3})$", fontsize="x-large")
ax3.plot(r_store[1:] / eta, S2_k_comp[1:], "b", label="Numerical result")
if "b" in keys_state_phys:
    ax3.plot(r_store[1:] / eta, S2_p_comp[1:], "g", label="$S_2^P$")
ax3.plot(r_store[1:] / eta, S2_k_th[1:], "r--", label="22/3 theoretical")
ax3.plot(r_store[1:] / eta, EK_array[1:], "k--", label=r"$E_K$")
ax3.set_title(
    f"$S_2^K(r)/(r^{{2/3}}\\epsilon^{{2/3}})$, {title}",
    fontsize="x-large",
)
ax3.set_xlabel("$r/\\eta$", fontsize="x-large")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.legend()
plt.tight_layout()
# if save:
#   plt.savefig("S2k_r_comp.png", dpi=300)

```
### J(r) and div(J)(r)

Plot $J(r)$:

```{code-cell} ipython3
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_ylabel(r"$-J_{KL}(r)/r\epsilon$", fontsize="x-large")
ax1.plot(r_store[1:] / eta, Jl_k_comp[1:], "b", label="Numerical result")
if "b" in keys_state_phys:
    ax1.plot(r_store[1:] / eta, Jl_p_comp[1:], "g", label="$J_P$")
    ax1.plot(r_store[1:] / eta, Jl_k_comp[1:] + Jl_p_comp[1:], "g", label="$J_P$")
ax1.plot(r_store[1:] / eta, Jl_k_th[1:], "r--", label="4/3 theoretical")
ax1.set_title(f"$-J_{{KL}}(r)/r\\epsilon$, {title}", fontsize="x-large")
ax1.set_xlabel("$r/\\eta$", fontsize="x-large")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend()
plt.tight_layout()
# if save:
#    plt.savefig("Jk_r_compensate.png", dpi=300)

```

Plot $\nabla \cdot \mathbf{J}(r)$:

```{code-cell} ipython3
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_ylabel(r"$-\nabla \cdot J_{KL}(r)/4\epsilon$", fontsize="x-large")
ax2.plot(
    r_store[1:] / eta, -divJ_k[1:] / 4, "b", label="Numerical result"
)
if "b" in keys_state_phys:
    ax2.plot(
        r_store[1:] / eta,
        -divJ_p[1:] / 4,
        "g",
        label="$\\nabla \\cdot J_P$",
    )
ax2.plot(
    r_store[1:] / eta,
    np.ones_like(r_store[1:]),
    "r--",
    label="1 theoretical",
)
ax2.set_title(
    f"$-\\nabla \\cdot J_{{KL}}(r)/4\\epsilon$, {title}",
    fontsize="x-large",
)
ax2.set_xlabel("$r/\\eta$", fontsize="x-large")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.legend()
plt.tight_layout()
# if save:
#   plt.savefig("divJk_r_comp.png", dpi=300)

```

## J(rh, rv)

Plot $J(rh, rv)$:

```{code-cell} ipython3
fig1, ax1 = plt.subplots(figsize=(8, 6))
im = ax1.pcolormesh(
    RH[1:] / eta,
    RV[1:] / eta,
    -Jk_l_comp[1:],
    cmap="Blues",
    vmin=0.0,
    vmax=1.33,
)
fig1.colorbar(im, ax=ax1)
ax1.set_xlabel(r"$r_h/\eta$", fontsize="x-large")
ax1.set_ylabel(r"$r_v/\eta$", fontsize="x-large")
ax1.set_title(
    f"$-J_{{KL}}(r_h,r_v)/r\\epsilon$, {title}", fontsize="x-large"
)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(xmin=1)
ax1.set_ylim(ymin=1)
plt.tight_layout()
# if save:
#    plt.savefig("Jk_l_hv.png", dpi=300)


if "b" in keys_state_phys:
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    im = ax3.pcolormesh(
        RH[1:] / eta,
        RV[1:] / eta,
        -Jp_l_comp[1:],
        cmap="Greens",
        vmin=0.0,
        vmax=1.33,
    )
    fig3.colorbar(im, ax=ax3)
    ax3.set_xlabel(r"$r_h/\eta$", fontsize="x-large")
    ax3.set_ylabel(r"$r_v/\eta$", fontsize="x-large")
    ax3.set_title(
        f"$-J_{{PL}}(r_h,r_v)/r\\epsilon$, {title}",
        fontsize="x-large",
    )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(xmin=1)
    ax3.set_ylim(ymin=1)
    plt.tight_layout()
    # if save:
    #    plt.savefig("Jp_l_hv.png", dpi=300)
    
```

Plot $\nabla \cdot \mathbf{J}(rh, rv)$:
```{code-cell} ipython3
fig2, ax2 = plt.subplots(figsize=(8, 6))
im = ax2.pcolormesh(
    RH[1:] / eta,
    RV[1:] / eta,
    -divJk_hv[1:] / 4,
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
)
fig2.colorbar(im, ax=ax2)
ax2.set_xlabel(r"$r_h/\eta$", fontsize="x-large")
ax2.set_ylabel(r"$r_v/\eta$", fontsize="x-large")
ax2.set_title(
    f"$-\\nabla \\cdot J_{{KL}}(r_h,r_v)/4\\epsilon$, {title}",
    fontsize="x-large",
)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(xmin=1)
ax2.set_ylim(ymin=1)
plt.tight_layout()
# if save:
#    plt.savefig("divJk_hv.png", dpi=300)


if "b" in keys_state_phys:
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    im = ax4.pcolormesh(
        RH[1:] / eta,
        RV[1:] / eta,
        -divJp_hv[1:] / 4,
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
    )
    fig4.colorbar(im, ax=ax4)
    ax4.set_xlabel(r"$r_h/\eta$", fontsize="x-large")
    ax4.set_ylabel(r"$r_v/\eta$", fontsize="x-large")
    ax4.set_title(
        f"$-\\nabla \\cdot J_{{PL}}(r_h,r_v)/4\\epsilon$, {title}",
        fontsize="x-large",
    )
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlim(xmin=1)
    ax4.set_ylim(ymin=1)
    plt.tight_layout()
    # if save:
    #    plt.savefig("divJp_hv.png", dpi=300)
    
```


## Vectorial plot

Plot $\mathbf{J}$:

```{code-cell} ipython3
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_title(f"$-J_{{KL}}(r_h,r_v)$, {title}", fontsize="x-large")
ax1.quiver(RH, RV, -Jk_v, -Jk_h, width=0.005)
ax1.set_xlabel(r"$r_h$", fontsize="x-large")
ax1.set_ylabel(r"$r_v$", fontsize="x-large")
plt.tight_layout()
# if save:
#    plt.savefig("Jk_vector_hv.png", dpi=300)


# Normalized version
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_title(
    f"Normalized $-J_{{KL}}(r_h,r_v)$, {title}", fontsize="x-large"
)
RH_safe = np.where(RH != 0, RH, 1e-10)
RV_safe = np.where(RV != 0, RV, 1e-10)
ax2.quiver(RH, RV, -Jk_v / RV_safe, -Jk_h / RH_safe)
ax2.set_xlabel(r"$r_h$", fontsize="x-large")
ax2.set_ylabel(r"$r_v$", fontsize="x-large")
plt.tight_layout()
# if save:
#    plt.savefig("Jk_vector_hv_normalized.png", dpi=300)

plt.show()
```
