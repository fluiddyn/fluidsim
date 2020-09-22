import os
import argparse
import fluidsim
from fluidsim.base.params import Parameters
from fluidsim.util.util import available_solver_keys
import numpy as np


if os.getenv("SNIC_RESOURCE") is not None:
    # from fluiddyn.clusters.snic import ClusterSNIC as Cluster
    from fluiddyn.clusters.snic import Beskow36 as Cluster

    cluster_type = "snic"
else:
    from fluiddyn.clusters.local import ClusterLocal as Cluster

    cluster_type = "local"


def create_common_params(n0, n1=None, n2=None):
    params = Parameters("submit")
    params._set_attrib("weak", False)
    params._set_attrib("dry_run", False)
    params._set_attrib("mode", "")
    params._set_attrib("dim", 3)
    params._set_attrib("shape", "")
    params._set_attrib("output_dir", "")

    if n1 is None:
        n1 = n0

    params._set_child(
        "two_d",
        dict(
            shape=f"{n0} {n1}",
            time="00:20:00",
            solver="ns2d",
            fft_seq=["fft2d.with_fftw1d", "fft2d.with_fftw2d"],
            fft=["fft2d.mpi_with_fftw1d", "fft2d.mpi_with_fftwmpi2d"],
            nb_cores=np.array([2, 4, 8, 16, 32]),
            nodes=[],
        ),
    )

    if n2 is None:
        n2 = n0

    params._set_child(
        "three_d",
        dict(
            shape=f"{n0} {n1} {n2}",
            time="00:30:00",
            solver="ns3d",
            fft_seq=["fft3d.with_fftw3d"],
            fft=[
                "fft3d.mpi_with_fftw1d",
                "fft3d.mpi_with_fftwmpi3d",
                "fft3d.mpi_with_p3dfft",
                "fft3d.mpi_with_pfft",
            ],
            nb_cores=np.array([2, 4, 8, 16, 32]),
            nodes=[],
        ),
    )
    return params


def get_parser(prog="", description="", epilog=""):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("n0", nargs="?", type=int, default=None)
    parser.add_argument("n1", nargs="?", type=int, default=None)
    parser.add_argument("n2", nargs="?", type=int, default=None)
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default=None,
        help="Any of the following solver keys: {}".format(
            available_solver_keys()
        ),
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=3,
        help="dimension of the solver (default: 3)",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="simply print the commands which will be run",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="seq-intra-inter",
        help='could be "seq", "intra", "inter" or a combination of these',
    )
    parser.add_argument(
        "-nc",
        "--min-cores",
        type=int,
        default=1,
        help="min. no. of processes to use (default: 1)",
    )
    parser.add_argument(
        "-nn",
        "--min-nodes",
        type=int,
        default=2,
        help="max. no. of nodes to use for intra-node runs (default: 2)",
    )
    parser.add_argument(
        "-xn",
        "--max-nodes",
        type=int,
        default=2,
        help="max. no. of nodes to use for intra-node runs (default: 2)",
    )
    return parser


def parser_to_params(parser):
    args = parser.parse_args()
    if args.dim == 3:
        params = create_common_params(args.n0, args.n1, args.n2)
        params_dim = params.three_d
    else:
        params = create_common_params(args.n0, args.n1)
        params_dim = params.two_d

    if args.solver is not None:
        params_dim.solver = args.solver

    if args.min_cores > 1:
        log_min = np.log2(args.min_cores)
        params_dim.nb_cores = np.logspace(
            log_min, 10, int(10 - log_min) + 1, base=2, dtype=int
        )

    if args.max_nodes > 1:
        log_min = np.log2(args.min_nodes)
        log_max = np.log2(args.max_nodes)
        params_dim.nodes = np.logspace(
            log_min, log_max, int(log_max - log_min) + 1, base=2, dtype=int
        )

    params.dim = args.dim
    params.dry_run = args.dry_run
    params.mode = args.mode
    params.shape = params_dim.shape.replace(" ", "x")
    return params, params_dim


def init_cluster(params, Cluster, prefix="snic", subdir="benchmarks"):

    cluster = Cluster()
    if cluster.name_cluster == "beskow":
        cluster.default_project = "2017-12-20"
        cluster.nb_cores_per_node = 32
    else:
        cluster.default_project = "SNIC2017-12-20"

    output_dir = params.output_dir = os.path.abspath(
        "./../../fluidsim-bench-results/{}/{}_{}".format(
            subdir, cluster.name_cluster, params.shape
        )
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Output directory: ", output_dir)
    cluster.commands_unsetting_env.insert(0, "fluidinfo -o " + output_dir)
    return cluster


def submit(
    params,
    params_dim,
    cluster,
    nb_nodes,
    nb_cores_per_node=None,
    fft="all",
    cmd="bench",
):
    if nb_cores_per_node is None:
        nb_cores_per_node = cluster.nb_cores_per_node

    nb_mpi = nb_cores_per_node * nb_nodes
    nb_iterations = 20
    cmd = "fluidsim {} -s {} {} -t {} -it {} -o {}".format(
        cmd,
        params_dim.solver,
        params_dim.shape,
        fft,
        nb_iterations,
        params.output_dir,
    )
    if params.dry_run:
        print("np =", nb_mpi, end=" ")
        print(cmd)
    else:
        cluster.submit_command(
            cmd,
            name_run=f"{params_dim.solver}_{nb_mpi}",
            nb_nodes=nb_nodes,
            nb_cores_per_node=nb_cores_per_node,
            walltime=params_dim.time,
            nb_mpi_processes=nb_mpi,
            omp_num_threads=1,
            ask=False,
            bash=False,
            interactive=True,
            retain_script=True,
        )
