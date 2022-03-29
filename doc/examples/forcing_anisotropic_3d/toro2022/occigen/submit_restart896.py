import os
from pathlib import Path
import subprocess
import re

import pytimeparse
from capturer import CaptureOutput

from fluidsim.util import (
    times_start_last_from_path,
    get_last_estimated_remaining_duration,
)

from fluidoccigen import cluster

path_scratch = Path(os.environ["SCRATCHDIR"])

t_end = 30.0
nh = 896
nb_nodes = 4
nb_cores_per_node = 28
nb_mpi_processes = nb_nodes * nb_cores_per_node
max_elapsed_time = pytimeparse.parse("23:50:00")

path_simuls = sorted(
    (path_scratch / "aniso").glob(f"ns3d.strat_toro*_{nh}x{nh}*")
)


def run(command):
    return subprocess.run(
        command.split(), check=True, capture_output=True, text=True
    )


user = os.environ["USER"]
process = run(f"squeue -u {user}")
lines = process.stdout.split("\n")[1:]
jobs_id = [line.split()[0] for line in lines if line]

"""
Typical output of scontrol show job

JobId=12703083 JobName=N20.0_Rb40.0_896
   UserId=augier(3301) GroupId=egi2153(1765) MCS_label=N/A
   Priority=661130 Nice=0 Account=egi2153 QOS=petit
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=22:35:13 TimeLimit=1-00:00:00 TimeMin=1-00:00:00
   SubmitTime=2022-03-28T17:08:38 EligibleTime=2022-03-28T17:08:38
   AccrueTime=2022-03-28T17:08:38
   StartTime=2022-03-28T17:09:42 EndTime=2022-03-29T17:09:42 Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2022-03-28T17:09:42
   Partition=bdw28 AllocNode:Sid=login2:51898
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=n[3147,3157,3216,3983]
   BatchHost=n3147
   NumNodes=4 NumCPUs=224 NumTasks=112 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=224,mem=236000M,node=4,billing=224
   Socks/Node=* NtasksPerN:B:S:C=28:0:*:1 CoreSpec=*
   MinCPUsNode=28 MinMemoryNode=59000M MinTmpDiskNode=0
   Features=BDW28 DelayBoot=00:00:00
   OverSubscribe=NO Contiguous=0 Licenses=(null) Network=(null)
   Command=./launcher_2022-03-28_17-08-38.sh_5
   WorkDir=/panfs/panasas/cnt0022/egi2153/augier/Dev/fluidsim/doc/examples/forcing_anisotropic_3d/toro2022/occigen
   AdminComment=BDW28
   StdErr=/panfs/panasas/cnt0022/egi2153/augier/Dev/fluidsim/doc/examples/forcing_anisotropic_3d/toro2022/occigen/SLURM.N20.0_Rb40.0_896.%J.stderr
   StdIn=/dev/null
   StdOut=/panfs/panasas/cnt0022/egi2153/augier/Dev/fluidsim/doc/examples/forcing_anisotropic_3d/toro2022/occigen/SLURM.N20.0_Rb40.0_896.%J.stdout
   Power=
"""


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

for path_simul in path_simuls:
    t_start, t_last = times_start_last_from_path(path_simul)
    if t_last >= t_end:
        continue

    try:
        estimated_remaining_duration = get_last_estimated_remaining_duration(
            path_simul
        )
    except RuntimeError:
        print(
            "Cannot submit more jobs because no estimation of remaining duration"
        )
        continue

    N_str = re.search(r"_N(.*?)_", path_simul.name).group(1)
    N = float(N_str)
    Rb_str = re.search(r"_Rb(.*?)_", path_simul.name).group(1)
    Rb = float(Rb_str)

    name_run = f"N{N}_Rb{Rb}_{nh}"

    jobs_simul = {
        key: value for key, value in jobs_name.items() if value == name_run
    }
    jobs_runtime_simul = {key: jobs_runtime[key] for key in jobs_simul}

    print(f"{path_simul.name}: {t_last = }, {estimated_remaining_duration = }")
    print(f"{jobs_simul}")

    # convert in seconds
    estimated_remaining_duration = pytimeparse.parse(estimated_remaining_duration)

    print(f"{estimated_remaining_duration = }")

    command = f"fluidsim-restart {path_simul} --modify-params 'params.output.periods_print.print_stdout = 0.02'"

    # get last job id
    try:
        job_id = max(jobs_simul.keys())
    except ValueError:
        job_id = None

    # get second remaining from jobs
    time_submitted = max_elapsed_time * len(jobs_simul) - sum(
        pytimeparse.parse(runtime) for runtime in jobs_runtime_simul.values()
    )
    print(f"{time_submitted = }")

    if estimated_remaining_duration < time_submitted:
        continue
    # get number of jobs to be sent
    nb_jobs = (
        estimated_remaining_duration - time_submitted
    ) // max_elapsed_time + 1
    print(f"{nb_jobs} jobs need to be submitted")

    for i_job in range(nb_jobs):
        with CaptureOutput() as capturer:
            cluster.submit_command(
                command,
                name_run=name_run,
                nb_nodes=nb_nodes,
                nb_cores_per_node=nb_cores_per_node,
                nb_mpi_processes=nb_mpi_processes,
                omp_num_threads=1,
                ask=False,
                walltime="23:59:58",
                dependency=job_id,
            )
            text = capturer.get_text()

        job_id = text.split()[-1].strip()
