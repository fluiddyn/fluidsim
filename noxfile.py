import os
from datetime import timedelta
from pathlib import Path
from shutil import rmtree
from time import time

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})
nox.options.reuse_existing_virtualenvs = True


@nox.session
def validate_code(session):
    session.run_install(
        "pdm", "sync", "--clean", "-G", "dev", "--no-self", external=True
    )
    session.run("pdm", "validate_code", external=True)


def test_mpi_fft_lib(session, method_fft, nprocs=2, _k_expr=None, env=None):
    cmd = (
        f"mpirun -np {nprocs} --oversubscribe "
        "coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py"
    ).split()
    if _k_expr is not None:
        cmd.extend(["-k", _k_expr])

    if env is None:
        env = {}
    else:
        env = env.copy()
    env.update({"TRANSONIC_NO_REPLACE": "1", "FLUIDSIM_TYPE_FFT3D": method_fft})

    print(f"test for method {method_fft}")
    session.run(*cmd, external=True, env=env)


def _test(session, env=None, with_fft=True):
    path_coverage = Path(".coverage")
    rmtree(path_coverage, ignore_errors=True)
    path_coverage.mkdir(exist_ok=True)

    if env is not None:
        print(env)

    short_names = []
    if with_fft:
        short_names.extend(["fftwmpi3d", "fftw1d"])
        if "GITLAB_CI" in os.environ:
            short_names.append("pfft")

    for short_name in short_names:
        test_mpi_fft_lib(session, f"fft3d.mpi_with_{short_name}")

    if "GITLAB_CI" in os.environ and with_fft:
        test_mpi_fft_lib(session, "fft3d.mpi_with_pfft", nprocs=4)
        # tests with p3dfft cannot be run together...
        method = "fft3d.mpi_with_p3dfft"
        test_mpi_fft_lib(session, method, nprocs=2, _k_expr="not TestCoarse")
        test_mpi_fft_lib(session, method, nprocs=2, _k_expr="TestCoarse")
        test_mpi_fft_lib(session, method, nprocs=4, _k_expr="TestCoarse")

    cmd = "coverage run -p -m fluidsim.util.testing -v"
    for _env in ({}, {"TRANSONIC_NO_REPLACE": "1"}):
        if env is not None:
            _env.update(env)
        session.run(*cmd.split(), env=_env)

    cmd = "mpirun -np 2 --oversubscribe coverage run -p -m fluidsim.util.testing -v --exitfirst"
    session.run(*cmd.split(), env=env, external=True)

    session.run("coverage", "combine")
    session.run("coverage", "report")
    session.run("coverage", "xml")


@nox.session
def test_without_fft_and_pythran(session):
    command = "pdm sync --clean -G dev -G test -G mpi --no-self"
    session.run_install(*command.split(), external=True)
    session.install(
        ".", "-C", "setup-args=-Dtransonic-backend=python", "--no-deps"
    )

    _test(
        session,
        env={"TRANSONIC_BACKEND": "python", "TRANSONIC_NO_REPLACE": "1"},
        with_fft=False,
    )


class TimePrinter:
    def __init__(self):
        self.time_start = self.time_last = time()

    def __call__(self, task: str):
        time_now = time()
        if self.time_start != self.time_last:
            print(
                f"Time for {task}: {timedelta(seconds=time_now - self.time_last)}"
            )
        print(
            f"Session started since {timedelta(seconds=time_now - self.time_start)}"
        )
        self.time_last = time_now


@nox.session
def test_with_fft_and_pythran(session):
    print_times = TimePrinter()

    command = "pdm sync --clean -G dev -G test -G fft -G mpi --no-self"
    session.run_install(*command.split(), external=True)

    print_times("pdm sync")

    command = ". -v --no-deps -C setup-args=-Dnative=true"
    if "GITLAB_CI" in os.environ:
        command += " -C compile-args=-j1"
    session.install(*command.split())

    print_times("installing fluidsim")

    short_names = ["fftw", "mpi_with_fftw", "fftwmpi"]
    if "GITLAB_CI" in os.environ:
        short_names.extend(["pfft", "p3dfft"])
    for short_name in short_names:
        session.run_install("pip", "install", f"fluidfft-{short_name}", "-v")

    _test(session)

    print_times("tests")


@nox.session(name="test-examples")
def test_examples(session):
    """Execute the examples using pytest"""

    command = "pdm sync --clean -G test -G mpi -G fft -G dev --no-self"
    session.run_install(*command.split(), external=True)

    command = "."
    if "GITLAB_CI" in os.environ:
        command += " -C compile-args=-j1"
    session.install(*command.split())
    session.install("fluidfft-fftwmpi")

    session.chdir("doc/examples")
    session.run("make", "test", external=True)
    session.run("make", "test_mpi", external=True)


@nox.session
def doc(session):
    """Build the documentation"""
    print_times = TimePrinter()
    command = "pdm sync -G doc -G fft -G test -G dev --no-self"
    session.run_install(*command.split(), external=True)
    print_times("pdm sync")

    session.install(
        ".", "-C", "setup-args=-Dtransonic-backend=python", "--no-deps"
    )
    print_times("install self")

    session.chdir("doc")
    session.run("make", "cleanall", external=True)
    session.run("make", external=True)
    print_times("make doc")


def _get_version_from_pyproject(path=Path.cwd()):
    if isinstance(path, str):
        path = Path(path)

    if path.name != "pyproject.toml":
        path /= "pyproject.toml"

    in_project = False
    version = None
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.startswith("[project]"):
                in_project = True
            if line.startswith("version =") and in_project:
                version = line.split("=")[1].strip()
                version = version[1:-1]
                break

    assert version is not None
    return version


@nox.session(name="add-tag-for-release", venv_backend="none")
def add_tag_for_release(session):
    session.run("hg", "pull", external=True)

    result = session.run(
        *"hg log -r default -G".split(), external=True, silent=True
    )
    if result[0] != "@":
        session.run("hg", "update", "default", external=True)

    version = _get_version_from_pyproject()
    version_core = _get_version_from_pyproject("lib")

    print(f"{version = }, {version_core = }")
    if version != version_core:
        session.error("version != version_core")

    result = session.run("hg", "tags", "-T", "{tag},", external=True, silent=True)
    last_tag = result.split(",", 2)[1]
    print(f"{last_tag = }")

    if last_tag == version:
        session.error("last_tag == version")

    answer = input(
        f'Do you really want to add and push the new tag "{version}"? (yes/[no]) '
    )

    if answer != "yes":
        print("Maybe next time then. Bye!")
        return

    print("Let's go!")
    session.run("hg", "tag", version, external=True)
    session.run("hg", "push", external=True)


@nox.session(python=False)
def detect_pythran_extensions(session):
    """Detect and print Pythran extension modules"""
    begin = "- "
    # begin = "import "
    paths_pythran_files = sorted(Path("fluidsim").rglob("*/__pythran__/*.py"))
    print(
        begin
        + f"\n{begin}".join(
            [str(p)[:-3].replace("/", ".") for p in paths_pythran_files]
        )
    )
