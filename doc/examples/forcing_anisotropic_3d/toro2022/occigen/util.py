import os
import subprocess
from pathlib import Path
from itertools import product

path_scratch = Path(os.environ["SCRATCHDIR"])


def run(command):
    return subprocess.run(
        command.split(), check=True, capture_output=True, text=True
    )


user = os.environ["USER"]


def get_info_jobs():

    process = run(f"squeue -u {user}")
    lines = process.stdout.split("\n")[1:]
    jobs_id = [line.split()[0] for line in lines if line]

    jobs_name = {}
    jobs_runtime = {}
    for job_id in jobs_id:
        process = run(f"scontrol show job {job_id}")
        for line in process.stdout.split("\n"):
            line = line.strip()
            if line.startswith("JobId"):
                job_name = line.split(" JobName=")[1]
                jobs_name[job_id] = job_name
            elif line.startswith("RunTime"):
                jobs_runtime[job_id] = line.split("RunTime=")[1].split(" ")[0]

    return jobs_id, jobs_name, jobs_runtime


def lprod(a, b):
    return list(product(a, b))
