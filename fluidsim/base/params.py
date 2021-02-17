"""Information on a solver (:mod:`fluidsim.base.params`)
==============================================================


Provides:

.. autoclass:: Parameters
   :members:
   :private-members:


"""
from fluidsim_core.params import Parameters
from fluidsim.base.solvers.info_base import InfoSolverBase


def fix_old_params(params):
    """Fix old parameters with depreciated values."""
    # params.FORCING -> params.forcing.enable (2018-02-16)
    try:
        params.FORCING
    except AttributeError:
        pass
    else:
        try:
            params.forcing
        except AttributeError:
            pass
        else:
            params.forcing._set_attrib("enable", params.FORCING)

    # for ns2d.strat (wrong parameter params.NO_SHEAR_MODES)
    try:
        params.NO_SHEAR_MODES
    except AttributeError:
        pass
    else:
        try:
            params.oper.NO_SHEAR_MODES = params.NO_SHEAR_MODES
        except AttributeError:
            pass


def merge_params(to_params, *other_params):
    """Merges missing parameters attributes and children of a typical
    Simulation object's parameters when compared to other parameters.
    Also, tries to replace `to_params.oper.type_fft` if found to be
    not based on FluidFFT.

    Parameters
    ----------
    to_params: Parameters

    other_params: Parameters, Parameters, ...

    """
    for other in other_params:
        to_params |= other

    # Substitute old FFT types with newer FluidFFT implementations
    if hasattr(to_params, "oper") and hasattr(to_params.oper, "type_fft"):
        method = to_params.oper.type_fft
        if (
            not method
            or method != "default"
            and not any(
                [
                    method.startswith(prefix)
                    for prefix in ("fft2d.", "fft3d.", "fluidfft.")
                ]
            )
        ):
            type_fft = "default"
            print("params.oper.type_fft", to_params.oper.type_fft, "->", type_fft)
            to_params.oper.type_fft = type_fft


create_params = Parameters._create_params
load_params_simul = Parameters._load_params_simul
load_info_solver = Parameters._load_info_solver


if __name__ == "__main__":
    info_solver = InfoSolverBase(tag="solver")

    params = create_params(info_solver)

# info = create_info_simul(info_solver, params)
