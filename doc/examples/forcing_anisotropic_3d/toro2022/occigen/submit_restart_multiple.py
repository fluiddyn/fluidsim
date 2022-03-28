import sys

from fluidsim import FLUIDSIM_PATH
from fluidoccigen import cluster, Occigen


# cluster
job_id = 11939832
nb_jobs_added = 13

cluster.launch_more_dependant_jobs(job_id, nb_jobs_added, path_launcher=None)
