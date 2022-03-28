
from fluidoccigen import cluster

job_id = 11939832
nb_jobs_added = 1

cluster.launch_more_dependant_jobs(job_id, nb_jobs_added, path_launcher=None)
