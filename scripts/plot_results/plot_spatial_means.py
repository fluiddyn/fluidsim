import numpy as np
import matplotlib.pylab as plt

from solveq2d import solveq2d

name_dir = (
    "SE2D_SW1lwaves_forcingw_L=50.x50._256x256_c=50_f=0_2013-06-14_09-48-06"
)
sim = solveq2d.create_sim_plot_from_dir(name_dir)

nx = sim.param.nx
c2 = sim.param.c2
f = sim.param.f
name_solver = sim.output.name_solver

dict_results = sim.output.spatial_means.load()


def derivate1D(t, func):
    dt_func = np.empty(func.shape, dtype=func.dtype)
    dt_func[1:-1] = (func[2:] - func[0:-2]) / (t[2:] - t[0:-2])
    dt_func[0] = (func[1] - func[0]) / (t[1] - t[0])
    dt_func[-1] = (func[-1] - func[-2]) / (t[-1] - t[-2])
    return dt_func


t = dict_results["t"]

E = dict_results["E"]
CPE = dict_results["CPE"]

EK = dict_results["EK"]
EA = dict_results["EA"]
EKr = dict_results["EKr"]

epsK = dict_results["epsK"]
epsK_hypo = dict_results["epsK_hypo"]
epsK_tot = dict_results["epsK_tot"]

epsA = dict_results["epsA"]
epsA_hypo = dict_results["epsA_hypo"]
epsA_tot = dict_results["epsA_tot"]

epsE = epsK + epsA
epsE_hypo = epsK_hypo + epsA_hypo
epsE_tot = epsK_tot + epsA_tot

epsCPE = dict_results["epsCPE"]
epsCPE_hypo = dict_results["epsCPE_hypo"]
epsCPE_tot = dict_results["epsCPE_tot"]

PK_tot = dict_results["PK_tot"]
PA_tot = dict_results["PA_tot"]
P_tot = PK_tot + PA_tot


width_axe = 0.85
height_axe = 0.37
x_left_axe = 0.12
z_bottom_axe = 0.56

size_axe = [x_left_axe, z_bottom_axe, width_axe, height_axe]
fig, ax1 = sim.output.figure_axe(size_axe=size_axe)
ax1.set_xlabel("$t$")
ax1.set_ylabel("$2E(t)/c^2$")
title = (
    "mean energy, solver "
    + sim.output.name_solver
    + f", nh = {nx:5d}"
    + f", c2 = {c2:.4g}, f = {f:.4g}"
)

ax1.set_title(title)
ax1.hold(True)
norm = c2 / 2
ax1.plot(t, E / norm, "k", linewidth=2)
ax1.plot(t, EK / norm, "r", linewidth=1)
ax1.plot(t, EA / norm, "b", linewidth=1)
ax1.plot(t, EKr / norm, "r--", linewidth=1)
ax1.plot(t, (EK - EKr) / norm, "r:", linewidth=1)

z_bottom_axe = 0.07
size_axe[1] = z_bottom_axe
ax2 = fig.add_axes(size_axe)
ax2.set_xlabel("t")
ax2.set_ylabel("Charney PE(t)")
title = "mean Charney PE(t)"
ax2.set_title(title)
ax2.hold(True)
ax2.plot(t, CPE, "k", linewidth=2)


z_bottom_axe = 0.56
size_axe[1] = z_bottom_axe
fig, ax1 = sim.output.figure_axe(size_axe=size_axe)
ax1.set_xlabel("t")
ax1.set_ylabel(r"$P_E(t)$, $\epsilon(t)$")
title = (
    "forcing and dissipation, solver "
    + sim.output.name_solver
    + f", nh = {nx:5d}"
    + f", c2 = {c2:.4g}, f = {f:.4g}"
)
ax1.set_title(title)
ax1.hold(True)
(l_P_tot,) = ax1.plot(t, P_tot, "c", linewidth=2)
(l_epsE,) = ax1.plot(t, epsE, "k--", linewidth=2)
(l_epsE_hypo,) = ax1.plot(t, epsE_hypo, "g", linewidth=2)
(l_epsE_tot,) = ax1.plot(t, epsE_tot, "k", linewidth=2)
l_P_tot.set_label("$P_{tot}$")
l_epsE.set_label(r"$\epsilon$")
l_epsE_hypo.set_label(r"$\epsilon_{hypo}$")
l_epsE_tot.set_label(r"$\epsilon_{tot}$")

ax1.legend(loc=2)

z_bottom_axe = 0.07
size_axe[1] = z_bottom_axe
ax2 = fig.add_axes(size_axe)
ax2.set_xlabel("t")
ax2.set_ylabel(r"$\epsilon$ Charney PE(t)")
title = "dissipation Charney PE"
ax2.set_title(title)
ax2.hold(True)
ax2.plot(t, epsCPE, "k--", linewidth=2)
ax2.plot(t, epsCPE_hypo, "g", linewidth=2)
ax2.plot(t, epsCPE_tot, "r", linewidth=2)

if dict_results.has_key("epsKsuppl"):
    epsKsuppl = dict_results["epsKsuppl"]
    epsKsuppl_hypo = dict_results["epsKsuppl_hypo"]

    ax1.plot(t, epsKsuppl, "r--", linewidth=1)
    ax1.plot(t, epsKsuppl_hypo, "g--", linewidth=1)

if "Conv" in dict_results.keys():
    print("Conv is saved")
    Conv = dict_results["Conv"]
    c2eta1d = dict_results["c2eta1d"]
    c2eta2d = dict_results["c2eta2d"]
    c2eta3d = dict_results["c2eta3d"]

    Conv2 = c2eta1d + c2eta2d / 2

    eps_mean = epsE.mean()
    E_mean = E.mean()

    # fig, ax1 = sim.output.figure_axe()

    # title = ('Conv, solver '+sim.output.name_solver+
    #      ', nh = {0:5d}'.format(nx)+
    #      ', c2 = {0:.4g}, f = {1:.4g}'.format(c2, f)
    #          )
    # ax1.set_title(title)
    # ax1.set_xlabel('t')

    # ax1.plot(t, Conv/eps_mean, 'k', linewidth=2)
    # ax1.plot(t, ConvA/eps_mean, 'k--', linewidth=2)

    print("E_mean  =", E_mean)
    print("eps_mean  =", eps_mean)
    print("Conv.mean()/eps_mean  =", Conv.mean() / eps_mean)
    print("Conv2.mean()/eps_mean  =", Conv2.mean() / eps_mean)
    print("c2eta1d.mean()/eps_mean  =", c2eta1d.mean() / eps_mean)
    print("c2eta2d.mean()/eps_mean  =", c2eta2d.mean() / eps_mean)
    print("c2eta3d.mean()/eps_mean  =", c2eta3d.mean() / eps_mean)

    # norm = E_mean**2

    # ax1.plot(t, cetaConv/norm, 'r', linewidth=2)
    # ax1.plot(t, cetaConvA/norm, 'r--', linewidth=2)

    # print('cetaConv.mean()/norm  =', cetaConv.mean()/norm)
    # print('cetaConvA.mean()/norm =', cetaConvA.mean()/norm)

solveq2d.show()
