"""Utilities for the numerical simulations (:mod:`fluidsim.util`)
=================================================================

.. autofunction:: load_sim_for_plot

.. autofunction:: load_state_phys_file

"""

from typing import Union
import os
from copy import deepcopy as _deepcopy
import inspect
from pathlib import Path

import numpy as _np
import h5py as _h5py

from importlib import import_module


import fluiddyn as fld

from fluiddyn.util import mpi

from fluidsim import path_dir_results, solvers

from fluidsim.base.params import (
    load_info_solver,
    load_params_simul,
    Parameters,
    merge_params,
    fix_old_params,
)


def available_solver_keys(package=solvers):
    """Inspects a package or a subpackage for all available
    solvers.

    Returns
    -------
    list

    """
    if isinstance(package, str):
        package = import_module(package)

    top = os.path.split(inspect.getfile(package))[0]
    top = os.path.abspath(top) + os.sep
    keys = list()
    for dirpath, dirname, filenames in os.walk(top):
        if "solver.py" in filenames:
            dirpath = os.path.abspath(dirpath)
            key = dirpath.replace(top, "")
            key = key.replace(os.sep, ".")
            keys.append(key)

    return sorted(keys)


def _get_key_package(key, package):
    """Compute (key, package) from (key, package) with default value"""
    if package is None:
        if key.startswith("fluidsim"):
            package, key = key.split(".", 1)
        else:
            package = "fluidsim.solvers"
    return key, package


def module_solver_from_key(key=None, package=None):
    """Return the string corresponding to a module solver."""
    key = key.lower()
    key, package = _get_key_package(key, package)

    keys = available_solver_keys(package)

    if key in keys:
        part_path = key
    else:
        raise ValueError(
            "You have to give a proper solver key, name solver given: "
            "{}. Expected one of: {}".format(key, keys)
        )

    base_solvers = package
    module_solver = base_solvers + "." + part_path + ".solver"

    return module_solver


def import_module_solver_from_key(key=None, package=None):
    """Import and reload the solver.

    Parameters
    ----------

    key : str
        The short name of a solver.

    """
    key, package = _get_key_package(key, package)

    return import_module(module_solver_from_key(key, package))


def get_dim_from_solver_key(key, package=None):
    """Try to guess the dimension from the solver key (via the operator name).

    """
    cls = import_simul_class_from_key(key, package)
    info = cls.InfoSolver()
    class_name = info.classes.Operators.class_name

    # special cases:
    if class_name == "OperatorsPseudoSpectralSW1L":
        return "2"

    for dim in range(4):
        if str(dim) in class_name:
            return str(dim)

    raise NotImplementedError(
        "Cannot deduce dimension of the solver from the name " + class_name
    )


def import_simul_class_from_key(key, package=None):
    """Import and reload a simul class.

    Parameters
    ----------

    key : str
        The short name of a solver.

    """
    solver = import_module(module_solver_from_key(key, package))
    return solver.Simul


def pathdir_from_namedir(name_dir: Union[str, Path, None] = None):
    """Return the path of a result directory.

    name_dir: str, optional

      Can be an absolute path, a relative path, or even simply just
      the name of the directory under $FLUIDSIM_PATH.

    """
    if name_dir is None:
        return os.getcwd()

    if not isinstance(name_dir, Path):
        name_dir = Path(name_dir)

    name_dir = name_dir.expanduser()

    if name_dir.is_dir():
        return name_dir.absolute()

    if not name_dir.is_absolute():
        name_dir = path_dir_results / name_dir

    if not name_dir.is_dir():
        raise ValueError(str(name_dir) + " is not a directory")

    return name_dir


class ModulesSolvers(dict):
    """Dictionary to gather imported solvers."""

    def __init__(self, names_solvers, package=None):
        for key in names_solvers:
            self[key] = import_module_solver_from_key(key, package)


def name_file_from_time_approx(path_dir, t_approx=None):
    """Return the file name whose time is the closest to the given time.

    .. todo::

        Can be elegantly implemented using regex as done in
        ``fluidsim.base.output.phys_fields.time_from_path``

    """
    if not isinstance(path_dir, Path):
        path_dir = Path(path_dir)

    path_files = sorted(tuple(path_dir.glob("state_phys_t*")))
    nb_files = len(path_files)
    if nb_files == 0 and mpi.rank == 0:
        raise ValueError("No state file in the dir\n" + str(path_dir))

    name_files = [path.name for path in path_files]
    if "state_phys_t=" in name_files[0]:
        ind_start_time = len("state_phys_t=")
    else:
        ind_start_time = len("state_phys_t")

    times = _np.empty([nb_files])
    for ii, name in enumerate(name_files):
        tmp = ".".join(name[ind_start_time:].split(".")[:2])
        if "_" in tmp:
            tmp = tmp[: tmp.index("_")]
        times[ii] = float(tmp)
    if t_approx is None:
        t_approx = times.max()
    i_file = abs(times - t_approx).argmin()
    name_file = path_files[i_file].name
    return name_file


def load_sim_for_plot(name_dir=None, merge_missing_params=False):
    """Create a object Simul from a dir result.

    Creating simulation objects with this function should be fast because the
    state is not initialized with the output file and only a coarse operator is
    created.

    Parameters
    ----------

    name_dir : str (optional)

      Name of the directory of the simulation. If nothing is given, we load the
      data in the current directory.
      Can be an absolute path, a relative path, or even simply just
      the name of the directory under $FLUIDSIM_PATH.

    merge_missing_params : bool (optional, default == False)

      Can be used to load old simulations carried out with an old fluidsim
      version.

    """
    path_dir = pathdir_from_namedir(name_dir)
    solver = _import_solver_from_path(path_dir)
    params = load_params_simul(path_dir)

    if merge_missing_params:
        merge_params(params, solver.Simul.create_default_params())

    params.path_run = path_dir
    params.init_fields.type = "constant"
    params.init_fields.modif_after_init = False
    params.ONLY_COARSE_OPER = True
    params.NEW_DIR_RESULTS = False
    params.output.HAS_TO_SAVE = False
    params.output.ONLINE_PLOT_OK = False

    try:
        params.preprocess.enable = False
    except AttributeError:
        pass

    fix_old_params(params)

    sim = solver.Simul(params)
    return sim


def _import_solver_from_path(path_dir):
    info_solver = load_info_solver(path_dir)
    solver = import_module(info_solver.module_name)
    return solver


def load_state_phys_file(
    name_dir=None,
    t_approx=None,
    modif_save_params=True,
    merge_missing_params=False,
):
    """Create a simulation from a file.

    For large resolution, creating a simulation object with this function can
    be slow because the state is initialized with the output file.

    Parameters
    ----------

    name_dir : str (optional)

      Name of the directory of the simulation. If nothing is given, we load the
      data in the current directory.
      Can be an absolute path, a relative path, or even simply just
      the name of the directory under $FLUIDSIM_PATH.

    t_approx : number (optional)

      Approximate time of the file to be loaded.

    modif_save_params :  bool (optional, default == True)

      If True, the parameters of the simulation are modified before loading::

        params.output.HAS_TO_SAVE = False
        params.output.ONLINE_PLOT_OK = False

    merge_missing_params : bool (optional, default == False)

      Can be used to load old simulations carried out with an old fluidsim
      version.

    """

    params, Simul = load_for_restart(name_dir, t_approx, merge_missing_params)

    if modif_save_params:
        params.output.HAS_TO_SAVE = False
        params.output.ONLINE_PLOT_OK = False

    params.ONLY_COARSE_OPER = False

    sim = Simul(params)
    return sim


def load_for_restart(name_dir=None, t_approx=None, merge_missing_params=False):
    """Load params and Simul for a restart.

    Parameters
    ----------

    name_dir : str (optional)

      Name of the directory of the simulation. If nothing is given, we load the
      data in the current directory.
      Can be an absolute path, a relative path, or even simply just
      the name of the directory under $FLUIDSIM_PATH.

    t_approx : number (optional)

      Approximate time of the file to be loaded.

    merge_missing_params : bool (optional, default == False)

      Can be used to load old simulations carried out with an old fluidsim
      version.

    """

    path_dir = pathdir_from_namedir(name_dir)
    solver = _import_solver_from_path(path_dir)

    # choose the file with the time closer to t_approx
    name_file = name_file_from_time_approx(path_dir, t_approx)
    path_file = os.path.join(path_dir, name_file)

    if merge_missing_params:
        # this has to be done by all processes otherwise there is a problem
        # with Transonic (see https://bitbucket.org/fluiddyn/fluidsim/issues/26)
        default_params = solver.Simul.create_default_params()

    if mpi.rank > 0:
        params = None
    else:
        with _h5py.File(path_file, "r") as f:
            params = Parameters(hdf5_object=f["info_simul"]["params"])

        if merge_missing_params:
            merge_params(params, default_params)

        params.path_run = path_dir
        params.NEW_DIR_RESULTS = False
        params.init_fields.type = "from_file"
        params.init_fields.from_file.path = path_file
        params.init_fields.modif_after_init = False
        try:
            params.preprocess.enable = False
        except AttributeError:
            pass

        fix_old_params(params)

    if mpi.nb_proc > 1:
        params = mpi.comm.bcast(params, root=0)

    return params, solver.Simul


def modif_resolution_all_dir(t_approx=None, coef_modif_resol=2, dir_base=None):
    """Save files with a modified resolution."""
    raise DeprecationWarning("Sorry, use modif_resolution_from_dir instead")


def modif_resolution_from_dir(
    name_dir=None, t_approx=None, coef_modif_resol=2, PLOT=True
):
    """Save a file with a modified resolution."""

    path_dir = pathdir_from_namedir(name_dir)

    solver = _import_solver_from_path(path_dir)

    sim = load_state_phys_file(name_dir, t_approx)

    params2 = _deepcopy(sim.params)
    params2.oper.nx = int(sim.params.oper.nx * coef_modif_resol)
    params2.oper.ny = int(sim.params.oper.ny * coef_modif_resol)
    params2.init_fields.type = "from_simul"

    sim2 = solver.Simul(params2)
    sim2.init_fields.get_state_from_simul(sim)

    print(sim2.params.path_run)

    sim2.output.path_run = str(path_dir) + "/State_phys_{}x{}".format(
        sim2.params.oper.nx, sim2.params.oper.ny
    )
    print("Save file in directory\n" + sim2.output.path_run)
    sim2.output.phys_fields.save(particular_attr="modif_resolution")

    print("The new file is saved.")

    if PLOT:
        sim.output.phys_fields.plot(numfig=0)
        sim2.output.phys_fields.plot(numfig=1)
        fld.show()


def times_start_end_from_path(path):
    """Return the start and end times from a result directory path.

    """

    path_file = path + "/stdout.txt"
    if not os.path.exists(path_file):
        print("Given path does not exist:\n " + path)
        return 666, 666

    with open(path_file, "r") as file_stdout:

        line = ""
        while not line.startswith("it ="):
            line = file_stdout.readline()

        words = line.split()
        t_s = float(words[6])

        # in order to get the information at the end of the file,
        # we do not want to read the full file...
        file_stdout.seek(0, os.SEEK_END)  # go to the end
        nb_caract = file_stdout.tell()
        nb_caract_to_read = min(nb_caract, 1000)
        file_stdout.seek(file_stdout.tell() - nb_caract_to_read, os.SEEK_SET)
        while line != "":
            if line.startswith("it ="):
                line_it = line
            last_line = line
            line = file_stdout.readline()

        if last_line.startswith("save state_phys"):
            word = last_line.replace("=", " ").split()[-1]
            _, ext = os.path.splitext(word)
            t_e = float(word.replace(ext, ""))
        else:
            words = line_it.split()
            t_e = float(words[6])

    # print('t_s = {0:.3f}, t_e = {1:.3f}'.format(t_s, t_e))

    return t_s, t_e
