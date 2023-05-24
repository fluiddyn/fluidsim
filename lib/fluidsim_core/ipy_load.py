"""For command fluidsim-ipy-load

.. autofunction:: start_ipython_load_sim

"""

from textwrap import dedent


def start_ipython_load_sim(load_import="from fluidsim import load"):
    """Start IPython and load a simulation"""
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
    argv.append("; ".join(code.strip().split("\n")))
    start_ipython(argv=argv)
