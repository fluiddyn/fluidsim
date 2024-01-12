from importlib import metadata

__version__ = metadata.version(__package__)


__all__ = ["__version__", "get_local_version", "__about__"]

try:
    from pyfiglet import figlet_format

    __about__ = figlet_format("fluidsim", font="big")
except ImportError:
    __about__ = r"""
  __ _       _     _     _
 / _| |     (_)   | |   (_)
| |_| |_   _ _  __| |___ _ _ __ ___
|  _| | | | | |/ _` / __| | '_ ` _ \
| | | | |_| | | (_| \__ \ | | | | | |
|_| |_|\__,_|_|\__,_|___/_|_| |_| |_|
"""

__about__ = __about__.rstrip() + f"\n\n{28 * ' '} v. {__version__}\n"


def get_local_version():
    """Get a long "local" version."""

    return __version__
