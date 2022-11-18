"""For command fluidsim-ipy-load

.. autofunction:: start_ipython_load_sim

"""

from textwrap import dedent


def start_ipython_load_sim():
    """Start IPython and load a simulation"""
    from IPython import start_ipython

    argv = ["--matplotlib", "-i", "-c"]
    code = dedent(
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from fluidsim import load
        print("Loading simulation")
        sim = load()
        params = sim.params
        print("`sim`, `params`, `np` and `plt` variables are available")
    """
    )
    argv.append("; ".join(code.strip().split("\n")))
    start_ipython(argv=argv)
