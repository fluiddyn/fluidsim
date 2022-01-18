"""Script for simulations with the solver ns3d.strat and the forcing
tcrandom_anisotropic.

## Examples

```
python run_simul.py --only-print-params
python run_simul.py --only-print-params-as-code

python run_simul.py -F 0.3 --delta-F 0.1 --ratio-kfmin-kf 0.8 --ratio-kfmax-kf 1.5 -opf
python run_simul.py -F 1.0 --delta-F 0.1 --ratio-kfmin-kf 0.8 --ratio-kfmax-kf 1.5 -opf

mpirun -np 2 python run_simul.py

```

This script is designed to study stratified turbulence forced with an
anisotropic forcing in toroidal or poloidal modes.

The regime depends on the value of the horizontal Froude number Fh and buoyancy
Reynolds numbers R and R4:

- Fh = epsK / (Uh^2 N)
- R = epsK / (N^2 nu)
- R4 = epsK Uh^2 / (nu4 N^4)

Fh has to be very small to be in a strongly stratified regime. R and R4 has to
be "large" to be in a turbulent regime.

For this forcing, we fix the injection rate P (very close to epsK). We will
work at P = 1.0, such that N, nu and nu4 determine the non dimensional numbers.

Note that Uh is not directly fixed by the forcing but should be a function of
the other input parameters. Dimensionally, we can write Uh = (P Lfh)**(1/3).

For simplicity, we'd like to have Lfh=1.0. We want to force at "large
horizontal scale" (compared to the size of the numerical domain). This length
(params.oper.Lx = params.oper.Ly) is computed with this condition.

"""

from util import main

if __name__ == "__main__":

    params, sim = main()
