from pathlib import Path

from transonic.dist import ParallelBuildExt

here = Path(__file__).parent.absolute()

from setup_configure import PARALLEL_COMPILE


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
