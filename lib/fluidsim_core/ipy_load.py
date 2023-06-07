"""For command fluidsim-ipy-load

.. autofunction:: start_ipython_load_sim

"""

import argparse
from textwrap import dedent


def start_ipython_load_sim(load_import="from fluidsim import load"):
    """Start IPython and load a simulation"""

    parser = argparse.ArgumentParser(
        prog="ipy-load",
        description="Start IPython and load a simulation",
    )
    parser.add_argument("path_dir", nargs="?", default=None)
    args = parser.parse_args()

    from IPython import start_ipython

    argv = ["--matplotlib", "-i", "-c"]

    code = dedent(
        f"""
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        {load_import}
        print("Loading simulation")
        sim = load()
        params = sim.params
        print("`sim`, `params`, `np`, `plt` and `pd` variables are available")
    """
    )
    lines = code.strip().split("\n")

    if args.path_dir is not None:
        from fluidsim_core.paths import find_path_result_dir

        path_dir = find_path_result_dir(args.path_dir)
        lines.insert(0, f"import os; os.chdir('{path_dir}')")

    argv.append("; ".join(lines))
    start_ipython(argv=argv)
