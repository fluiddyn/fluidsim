"""
Module where the version is written.

It is executed in setup.py and imported in fluidsim/__init__.py.

See:

http://en.wikipedia.org/wiki/Software_versioning
http://legacy.python.org/dev/peps/pep-0386/

'a' or 'alpha' means alpha version (internal testing),
'b' or 'beta' means beta version (external testing).

PEP 440 also permits the use of local version identifiers. This is initialized
using the setuptools_scm module, if available.

See:
https://www.python.org/dev/peps/pep-0440/#local-version-identifiers
https://github.com/pypa/setuptools_scm#setuptools_scm

"""


try:
    from setuptools_scm import get_version

    __version__ = get_version()
except:
    __version__ = "0.2.0"


try:
    from pyfiglet import figlet_format

    __about__ = figlet_format("fluidsim v" + __version__, font="big")
except:
    __about__ = r"""
  __ _       _     _     _
 / _| |     (_)   | |   (_)
| |_| |_   _ _  __| |___ _ _ __ ___
|  _| | | | | |/ _` / __| | '_ ` _ \
| | | | |_| | | (_| \__ \ | | | | | |
|_| |_|\__,_|_|\__,_|___/_|_| |_| |_|

"""
