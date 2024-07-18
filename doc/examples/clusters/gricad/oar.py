"""
OAR clusters (:mod:`fluiddyn.clusters.oar`)
===========================================

Provides:

.. autoclass:: ClusterOAR
   :members:

.. autofunction:: get_job_id

.. autofunction:: get_job_info

.. autofunction:: count_number_jobs
"""

import datetime
import os
import stat
import time
from subprocess import getoutput

from fluiddyn.io.query import call_bash, run_asking_agreement
from fluiddyn.clusters import Cluster, subprocess


def get_job_id(name_job):
    """Get the job id from the name job

    Typical output of ``oarstat -u``

    ::

      Job id    S User     Duration   System message
      --------- - -------- ---------- ------------------------------------------------
      644829    F augier3p    0:05:08 R=4,W=0:10:0,J=B,N=fluidsim-restart_nx160_Rb20_N20,T=idempotent (Karma=0.038,quota_ok)

    """
    output = getoutput("oarstat -u")
    if name_job not in output:
        return
    for line in output.split("\n"):
        if name_job in line:
            break
    return line.split(maxsplit=1)[0].strip()


def get_job_info(name_or_id):
    """Get some information about a job"""
    output = getoutput("oarstat -u")

    if not isinstance(name_or_id, str):
        name_or_id = str(name_or_id)

    if name_or_id not in output:
        return

    for line in output.split("\n"):
        if name_or_id in line:
            break

    keys = ["id", "status", "user", "duration", "message", "karma"]
    info = {k: w for k, w in zip(keys, line.split())}

    possible_status = {"F": "failing", "W": "waiting", "R": "running"}
    info["status"] = possible_status.get(info["status"], "?")

    keys_message = {"R": "nb_procs", "W": "walltime", "N": "name", "T": "type"}
    for param in info["message"].split(","):
        key, value = param.split("=")
        if key in keys_message:
            info[keys_message[key]] = value

    return info


def count_number_jobs(name_job):
    """Get the number of jobs returned by ``oarstat -u``"""
    output = getoutput("oarstat -u")
    return sum(True for line in output.split("\n") if name_job in line)


class ClusterOAR(Cluster):
    name_cluster = ""
    nb_cores_per_node = 12
    has_to_add_name_cluster = False

    _doc_commands = """
Useful commands
---------------
oarsub -S script.sh
oarstat -u
oardel $JOB_ID
oarsub -C $JOB_ID"""

    def check_oar(self):
        """check if this script is run on a frontal with oar installed"""
        try:
            subprocess.check_call(["oarsub", "--version"], stdout=subprocess.PIPE)
        except OSError as error:
            raise OSError("oar does not seem to be installed.") from error

    def submit_script(
        self,
        path,
        name_run="fluiddyn",
        nb_nodes=1,
        nb_cores_per_node=None,
        walltime="24:00:00",
        project=None,
        nb_mpi_processes=None,
        omp_num_threads=None,
        idempotent=False,
        anterior=None,
        delay_signal_walltime=300,
        network_address=None,
        ask=True,
        submit=True,
        run_with_exec=True,
        resource_conditions=None,
        use_oar_envsh=None,
    ):
        path = os.path.expanduser(path)
        if not os.path.exists(path.split(" ")[0]):
            raise ValueError("The script does not exists! path:\n" + path)

        if not path.startswith("python "):
            command = "python " + path

        self.submit_command(
            command,
            name_run=name_run,
            nb_nodes=nb_nodes,
            nb_cores_per_node=nb_cores_per_node,
            walltime=walltime,
            project=project,
            nb_mpi_processes=nb_mpi_processes,
            omp_num_threads=omp_num_threads,
            idempotent=idempotent,
            anterior=anterior,
            delay_signal_walltime=delay_signal_walltime,
            network_address=network_address,
            ask=ask,
            submit=submit,
            run_with_exec=run_with_exec,
            resource_conditions=resource_conditions,
            use_oar_envsh=use_oar_envsh,
        )

    def submit_command(
        self,
        command,
        name_run="fluiddyn",
        nb_nodes=1,
        nb_cores_per_node=None,
        walltime="24:00:00",
        project=None,
        nb_mpi_processes=None,
        omp_num_threads=None,
        idempotent=False,
        anterior=None,
        delay_signal_walltime=300,
        network_address=None,
        ask=True,
        submit=True,
        run_with_exec=True,
        resource_conditions=None,
        use_oar_envsh=None,
        devel=False,
    ):
        self.check_oar()

        nb_cores_per_node, nb_mpi_processes = self._parse_cores_procs(
            nb_nodes, nb_cores_per_node, nb_mpi_processes
        )

        str_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path_launching_script = "oar_launcher_" + str_time
        if os.path.exists(path_launching_script):
            i = 1
            while os.path.exists(path_launching_script + "_" + str(i)):
                i += 1
            path_launching_script += "_" + str(i)

        txt = self._create_txt_launching_script(
            command,
            name_run,
            nb_nodes,
            nb_cores_per_node,
            walltime,
            nb_mpi_processes=nb_mpi_processes,
            omp_num_threads=omp_num_threads,
            network_address=network_address,
            run_with_exec=run_with_exec,
            resource_conditions=resource_conditions,
            use_oar_envsh=use_oar_envsh,
            devel=devel,
            project=project,
        )

        with open(path_launching_script, "w", encoding="utf-8") as file:
            file.write(txt)

        os.chmod(
            path_launching_script, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
        )

        launching_command = "oarsub"

        if project is not None:
            launching_command += " --project " + project

        if delay_signal_walltime is not None:
            launching_command += " --checkpoint " + str(delay_signal_walltime)

        if idempotent:
            launching_command += " -t idempotent"

        if anterior is not None:
            launching_command += f" --anterior {anterior}"

        launching_command += " -S ./" + path_launching_script

        print(f'A launcher for the command "{command}" has been created.')
        if submit:
            if ask:
                run_asking_agreement(launching_command)
            else:
                print(
                    "The script is submitted with the command:\n",
                    launching_command,
                )
                call_bash(launching_command)

    def _create_txt_launching_script(
        self,
        command,
        name_run,
        nb_nodes,
        nb_cores_per_node,
        walltime,
        nb_mpi_processes=None,
        omp_num_threads=None,
        network_address=None,
        run_with_exec=True,
        resource_conditions=None,
        use_oar_envsh=None,
        devel=False,
        project=None,
    ):
        txt = f"#!/bin/bash\n\n#OAR -n {name_run}\n"

        if devel:
            txt += "#OAR -t devel\n"

        if project is not None:
            txt += f"#OAR --project {project}\n"

        txt += "#OAR -l "

        if self.has_to_add_name_cluster and network_address is None:
            conditions = f"cluster='{self.name_cluster}'"
        elif network_address is not None:
            conditions = f"network_address='{network_address}"
        else:
            conditions = ""

        if (
            hasattr(self, "resource_conditions")
            and self.resource_conditions is not None
        ):
            if resource_conditions is not None:
                resource_conditions = (
                    self.resource_conditions + " and " + resource_conditions
                )
            else:
                resource_conditions = self.resource_conditions

        if resource_conditions is not None:
            if conditions:
                conditions += " and "
            conditions += resource_conditions

        if conditions:
            txt += "{" + conditions + "}"

        txt += "/nodes={}/core={},walltime={}\n\n".format(
            nb_nodes, nb_cores_per_node, walltime
        )

        txt += 'echo "hostname: "$HOSTNAME\n\n'

        txt += "\n".join(self.get_commands_setting_env()) + "\n\n"

        if omp_num_threads is not None:
            txt += f"export OMP_NUM_THREADS={omp_num_threads}\n\n"

        if use_oar_envsh is None and hasattr(self, "use_oar_envsh"):
            use_oar_envsh = self.use_oar_envsh

        if use_oar_envsh is None:
            use_oar_envsh = nb_mpi_processes is not None and nb_nodes > 1

        if use_oar_envsh:
            txt += (
                "# Shell with environment variables forwarded\n"
                "export OMPI_MCA_plm_rsh_agent=oar-envsh\n\n"
            )

        if run_with_exec:
            txt += "exec "

        if hasattr(self, "after_exec"):
            txt += self.after_exec

        if nb_mpi_processes is not None:
            txt += f"mpirun -np {nb_mpi_processes} "

            if nb_nodes > 1:
                txt += "-machinefile $OAR_NODEFILE "

        txt += command + "\n"

        return txt

    def stall(self, name_job, limit_number_jobs=1, time_check=30):
        """Wait until job(s) completion.

        Parameters
        ----------

        name_run: str
            Description of the job. Should be the same as in submit_script.

        limit_number_jobs: int (default: 1)
            Stall when the number of job is larger or equal to `limit_number_jobs`.

        time_check: int
            Time to sleep in seconds.
        """
        tstart = time.time()
        while count_number_jobs(name_job) >= limit_number_jobs:
            time.sleep(time_check)
        print(f"job {name_job} finished in {time.time() - tstart} s")
