import numpy as np
import matplotlib.pylab as plt

from solveq2d import solveq2d


def jump_var_delta(sim, var, deltar):
    nx = sim.param.nx
    dx = sim.param.Lx / nx
    idelta = int(round(deltar / dx))
    jump = -var
    # jump = np.zeros(var.shape)
    jump[:, 0 : nx - idelta] += var[:, idelta:nx]
    jump[:, nx - idelta : nx] += var[:, 0:idelta]
    return jump


name_dir = (
    "/scratch/augier"
    # '/home/pierre'
    "/Results_SW1lw"
    "/Pure_standing_waves_512x512/"
    # 'SE2D_SW1lwaves_forcingw_L=50.x50._512x512_c=10_f=0_2013-06-12_00-02-04'
    # 'SE2D_SW1lwaves_forcingw_L=50.x50._512x512_c=200_f=0_2013-06-12_00-39-17'
    "SE2D_SW1lwaves_forcingw_L=50.x50._512x512_c=700_f=0_2013-06-12_12-40-31"
    # '/Pure_standing_waves_1024x1024/'
    # '/Pure_standing_waves_512x512/'
    # 'SE2D_SW1lwaves_forcingw_L=50.x50._1024x1024_c=10_f=0_2013-06-12_19-34-30'
    # 'SE2D_SW1lwaves_forcingw_L=50.x50._1024x1024_c=20_f=0_2013-06-12_20-33-19'
    # 'SE2D_SW1lwaves_forcingw_L=50.x50._1024x1024_c=50_f=0_2013-06-12_21-35-00'
    # 'SE2D_SW1lwaves_forcingw_L=50.x50._1024x1024_c=100_f=0_2013-06-12_22-44-21'
    # 'SE2D_SW1lwaves_forcingw_L=50.x50._1024x1024_c=200_f=0_2013-06-13_00-33-41'
    # 'SE2D_SW1lwaves_forcingw_L=50.x50._1024x1024_c=400_f=0_2013-06-13_03-42-31'
    # 'SE2D_SW1lwaves_forcingw_L=50.x50._1024x1024_c=700_f=0_2013-06-13_09-32-03'
    # 'SE2D_SW1lwaves_forcingw_L=50.x50._1024x1024_c=1000_f=0_2013-06-13_19-31-10'
)

deltar = 1.0
divJ_limit = 1

sim = solveq2d.load_state_phys_file(t_approx=262, name_dir=name_dir)

nx = sim.param.nx
c2 = sim.param.c2
c = np.sqrt(c2)
f = sim.param.f
name_solver = sim.output.name_solver


# sim.output.phys_fields.plot(key_field='eta')
# sim.output.phys_fields.plot(key_field='div')
# sim.output.phys_fields.plot(key_field='Jx')
# sim.output.phys_fields.plot(key_field='Jy')

eta = sim.state.state_phys["eta"]
h = eta + 1

ux = sim.state.state_phys["ux"]
uy = sim.state.state_phys["uy"]

Jx = h * ux
Jy = h * uy

oper = sim.oper

Jx_fft = oper.fft2(Jx)
Jy_fft = oper.fft2(Jy)
divJ_fft = oper.divfft_from_vecfft(Jx_fft, Jy_fft)
divJ = oper.ifft2(divJ_fft)

# divJ[abs(divJ)<divJ_limit] = 0.


eta_fft = oper.fft2(eta)
px_h_fft, py_h_fft = oper.gradfft_from_fft(eta_fft)

px_h = oper.ifft2(px_h_fft)
py_h = oper.ifft2(py_h_fft)
gradh2 = px_h**2 + py_h**2

# sim.output.phys_fields.plot(field=gradh2)

gradh2_limit = 60.0 / c2

Usx = divJ * px_h / gradh2
Usy = divJ * py_h / gradh2

Usx[abs(gradh2) < gradh2_limit] = 0
Usy[abs(gradh2) < gradh2_limit] = 0

Us_over_c = np.sqrt(Usx**2 + Usy**2) / c


def limit_maxmin(var, minv, maxv):
    var[var > maxv] = maxv
    var[var < minv] = minv


maxMach = 1.5
minMach = 0.3

limit_maxmin(Us_over_c, 0.0, 2.0)

sim.output.phys_fields.plot(field=Us_over_c, key_field="Us/c", nb_contours=20)


Machx2 = (ux - Usx) ** 2 / c2
Machy2 = (uy - Usy) ** 2 / c2
Mach = np.sqrt(Machx2 + Machy2)

Mach[abs(gradh2) < gradh2_limit] = 0

limit_maxmin(Mach, minMach, maxMach)

sim.output.phys_fields.plot(
    field=Mach, key_field="local Mach number", nb_contours=20
)


x = sim.oper.x_seq
y = sim.oper.y_seq

y0 = 29.5

iy0 = np.argmin(abs(y - y0))

h0 = h[iy0]
Jx0 = Jx[iy0]
Jy0 = Jy[iy0]

# x_left_axe = 0.12
# z_bottom_axe = 0.56
# width_axe = 0.85
# height_axe = 0.37
# size_axe = [x_left_axe, z_bottom_axe,
#             width_axe, height_axe]
# fig, ax1 = sim.output.figure_axe(size_axe=size_axe)

# ax1.hold(True)
# ax1.set_xscale('linear')
# ax1.set_yscale('linear')
# ax1.set_xlabel('$x$')
# ax1.set_ylabel('$h$')
# title = ('profils, solver '+sim.output.name_solver+
# ', nh = {0:5d}'.format(nx)+
# ', c = {0:.4g}, f = {1:.4g}'.format(c, f)
# )
# ax1.set_title(title)
# ax1.hold(True)
# ax1.plot(x, h0, 'k')


# z_bottom_axe = 0.09
# size_axe[1] = z_bottom_axe
# ax2 = fig.add_axes(size_axe)
# ax2.set_xlabel('$x$')
# ax2.set_ylabel('$J_x$, $J_y$')
# ax2.plot(x, Jx0, 'r')
# ax2.plot(x, Jy0, 'b')

solveq2d.show()
