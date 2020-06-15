from copy import deepcopy

import numpy as np
from dask.distributed import Client, LocalCluster

from run_profile import parser, main

coef = 1

nxs = np.array([216, 160, 144]) // 2
nxs *= coef

args_default = parser.parse_args()
args_default.t_end = 0.1 / coef


futures = []


def submit(nx, coef_dealiasing, scheme):
    print(f"submit ", (nx, coef_dealiasing, scheme))

    # path_out = path_stdout / (scheme + "_" + str(nx))

    args = deepcopy(args_default)
    args.nx = nx
    args.coef_dealiasing = coef_dealiasing
    args.type_time_scheme = scheme
    futures.append(client.submit(main, args, pure=False))


if __name__ == "__main__":

    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)

    print("port services", client.scheduler_info()["services"])

    nx = nxs[0]
    coef_dealiasing = 2 / 3

    for scheme in ["RK4", "RK2"]:
        submit(nx, coef_dealiasing, scheme)

    coefficients = [0.9, 1.0]

    for nx, coef_dealiasing in zip(nxs[1:], coefficients):
        for scheme in ["RK2_phaseshift", "RK2_phaseshift_random"]:
            submit(nx, coef_dealiasing, scheme)

    client.gather(futures)
