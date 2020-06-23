"""

```
python -c "from which_params import compute; compute(216, 1)"

python run_profile.py -nx 216 -cd 0.6666666 --type_time_scheme RK4 --t_end 0.1
python run_profile.py -nx 216 -cd 0.6666666 --type_time_scheme RK2 --t_end 0.1

python run_profile.py -nx 162 -cd 1.0 --type_time_scheme RK2_phaseshift --t_end 0.1
python run_profile.py -nx 162 -cd 1.0 --type_time_scheme RK2_phaseshift_random --t_end 0.1

python run_profile.py -nx 144 -cd 0.9 --type_time_scheme RK2_phaseshift --t_end 0.1
python run_profile.py -nx 144 -cd 0.9 --type_time_scheme RK2_phaseshift_random --t_end 0.1
```

"""
from pathlib import Path
import sys

from fluiddyn.util import mpi
from fluiddyn.io.tee import MultiFile

from run_simul import parser, init_params, init_state, Simul

from fluidsim.util.console.profile import run_profile

parser.set_defaults(max_elapsed="10:00:00", t_end=1.0)


class Tee:
    def __init__(self, files):
        self.files = list(files)

    def __enter__(self):
        sys.stdout = MultiFile(self.files + [sys.stdout])

    def __exit__(self, *args):
        sys.stdout = sys.__stdout__


def main(args):
    params = init_params(args)
    params.time_stepping.max_elapsed = None
    params.output.HAS_TO_SAVE = False
    params.output.periods_print.print_stdout = args.t_end / 4
    sim = Simul(params)
    init_state(sim, args)
    Path("tmp_profile").mkdir(exist_ok=True)
    base_name_file = f"tmp_profile/{sim.name_run}."
    name_file_log = base_name_file + "log"
    with open(name_file_log, "w") as file:
        with Tee([file]):
            run_profile(sim, path_results=base_name_file + "pstats")


if __name__ == "__main__":
    args = parser.parse_args()
    mpi.printby0(args)
    main(args)
