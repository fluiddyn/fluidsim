from pathlib import Path
from runpy import run_path

from transonic.dist import ParallelBuildExt

here = Path(__file__).parent.absolute()

try:
    from setup_config import PARALLEL_COMPILE
except ImportError:
    # needed when there is already a module with the same name imported.
    setup_config = run_path(here / "setup_config.py")
    PARALLEL_COMPILE = setup_config["PARALLEL_COMPILE"]


class FluidSimBuildExt(ParallelBuildExt):
    def initialize_options(self):
        super().initialize_options()
        self.logger_name = "fluidsim"
        self.num_jobs_env_var = "FLUIDDYN_NUM_PROCS_BUILD"

    def get_num_jobs(self):
        if PARALLEL_COMPILE:
            return super().get_num_jobs()
        else:
            # Return None which would in turn retain the `self.parallel` in its
            # default value
            return None
