"""

```
python -c "from which_params import compute; compute(256, 1)"
python run_profile.py -nx 128 -cd 0.6666666 --type_time_scheme RK4
python run_profile.py -nx 86 -cd 1. --type_time_scheme RK2_phaseshift
```

"""

from fluiddyn.util import mpi

from run_simul import parser, init_params, init_state, Simul

from fluidsim.util.console.profile import run_profile

parser.set_defaults(max_elapsed="10:00:00", t_end=1.0)


def main(args):
    params = init_params(args)
    params.time_stepping.max_elapsed = None
    params.output.HAS_TO_SAVE = False
    params.output.periods_print.print_stdout = args.t_end / 4
    sim = Simul(params)
    init_state(sim)
    run_profile(sim, path_results=f"tmp_profile/{sim.name_run}.pstats")


run_profile(sim, path_results="tmp_profile")
