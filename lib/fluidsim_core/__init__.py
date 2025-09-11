"""
Pure-Python core library for FluidSim framework. This package provides
generic base classes and utilities to build new solvers.

.. autosummary::
   :toctree:

   params
   info
   solver
   output
   loader
   magic
   extend_simul
   hexa_files
   scripts
   paths

"""


def __getattr__(name):
    if name == "__version__":
        from importlib import metadata

        return metadata.version(__package__)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
