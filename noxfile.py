import os

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})
nox.options.reuse_existing_virtualenvs = 1


@nox.session
def validate_code(session):
    session.run_always(
        "pdm", "sync", "--clean", "-G", "dev", "--no-self", external=True
    )
    session.run("pdm", "validate_code", external=True)


@nox.session
def test_without_fft_and_pythran(session):
    command = "pdm sync --clean -G dev -G test -G mpi --no-self"
    session.run_always(*command.split(), external=True)
    session.install(
        ".", "--config-settings=setup-args=-Dtransonic-backend=python", "--no-deps"
    )
    session.run(
        "make",
        "_tests_coverage",
        external=True,
        env={"TRANSONIC_BACKEND": "python", "TRANSONIC_NO_REPLACE": "1"},
    )


@nox.session
def test_with_fft_and_pythran(session):
    # first install fluidfft without Pythran compilation
    session.install("fluidfft", env={"FLUIDFFT_TRANSONIC_BACKEND": "python"})

    command = "pdm sync --clean -G dev -G test -G fft -G mpi --no-self"
    session.run_always(*command.split(), external=True)
    session.install(".", "--no-deps", "-C", "setup-args=-Dnative=true")

    session.run("make", "_tests_coverage", external=True)
