"""Utilities for paths"""

from typing import Union
from pathlib import Path

from fluiddyn.io import FLUIDSIM_PATH

try:
    path_dir_results = Path(FLUIDSIM_PATH)
except TypeError:
    # to be able to import for transonic
    path_dir_results = None


def find_path_result_dir(thing: Union[str, Path, None] = None):
    """Return the path of a result directory.

    thing: str or Path, optional

      Can be an absolute path, a relative path, or even simply just
      the name of the directory under $FLUIDSIM_PATH.

    """
    if thing is None:
        return Path.cwd()

    if not isinstance(thing, Path):
        path = Path(thing)
    else:
        path = thing

    path = path.expanduser()

    if path.is_dir():
        return path.absolute()

    if not path.is_absolute():
        path = path_dir_results / path

    if not path.is_dir():
        raise ValueError(f"Cannot find a path corresponding to {thing}")

    return path
