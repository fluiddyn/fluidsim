import os
import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})


@nox.session
def validate_code(session):
    session.run_always("pdm", "install", "-G", "dev", "--no-self", external=True)
    session.run("pdm", "validate_code", external=True)


@nox.session
def test_without_fft_and_pythran(session):
    command = "pdm install -G dev -G  test -G mpi -G scipy --no-self"
    session.run_always(*command.split(), external=True)
    session.install(".", "--config-settings=setup-args=-Dtransonic-backend=python")
    session.run(
        "make",
        "_tests_coverage",
        external=True,
        env={"TRANSONIC_BACKEND": "python", "TRANSONIC_NO_REPLACE": "1"},
    )
