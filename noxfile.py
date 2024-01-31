import os
from datetime import timedelta
from pathlib import Path
from shutil import rmtree
from time import time

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})
nox.options.reuse_existing_virtualenvs = 1


@nox.session
def validate_code(session):
    session.run_always(
        "pdm", "sync", "--clean", "-G", "dev", "--no-self", external=True
    )
    session.run("pdm", "validate_code", external=True)


def _test(session, env=None):
    path_coverage = Path(".coverage")
    rmtree(path_coverage, ignore_errors=True)
    path_coverage.mkdir(exist_ok=True)

    if env is not None:
        print(env)

    session.run("make", "_tests_coverage", external=True, env=env)
    session.run("coverage", "combine")
    session.run("coverage", "report")
    session.run("coverage", "xml")


@nox.session
def test_without_fft_and_pythran(session):
    command = "pdm sync --clean -G dev -G test -G mpi --no-self"
    session.run_always(*command.split(), external=True)
    session.install(
        ".", "-C", "setup-args=-Dtransonic-backend=python", "--no-deps"
    )

    _test(session, env={"TRANSONIC_BACKEND": "python", "TRANSONIC_NO_REPLACE": "1"})


def _install_fluidfft(session):
    # first install fluidfft without Pythran compilation
    session.install(
        "fluidfft", "--no-deps", env={"FLUIDFFT_TRANSONIC_BACKEND": "python"}
    )


time_last = 0


@nox.session
def test_with_fft_and_pythran(session):
    global time_last
    time_start = time_last = time()

    def print_times(task: str):
        global time_last
        time_now = time()
        if time_start != time_last:
            print(f"Time for {task}: {timedelta(seconds=time_now - time_last)}")
        print(f"Session started since {timedelta(seconds=time_now - time_start)}")
        time_last = time_now

    _install_fluidfft(session)

    print_times("installing fluidfft")

    command = "pdm sync --clean -G dev -G test -G fft -G mpi --no-self"
    session.run_always(*command.split(), external=True)

    print_times("pdm sync")

    command = ". -v --no-deps -C setup-args=-Dnative=true"
    if "GITLAB_CI" in os.environ:
        command += " -C compile-args=-j2"
    session.install(*command.split())

    print_times("installing fluidsim")

    _test(session)

    print_times("tests")


@nox.session
def doc(session):
    _install_fluidfft(session)
    command = "pdm sync -G doc -G fft -G test --no-self"
    session.run_always(*command.split(), external=True)
    session.install(".", "-C", "setup-args=-Dtransonic-backend=python", "--no-deps")

    session.chdir("doc")
    session.run("make", "cleanall", external=True)
    session.run("make", external=True)


def _get_version_from_pyproject(path=Path.cwd()):
    if isinstance(path, str):
        path = Path(path)

    if not path.name == "pyproject.toml":
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

    result = session.run(*"hg log -r default -G".split(), external=True, silent=True)
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
