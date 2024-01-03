import os
import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})


@nox.session
def validate_code(session):
    session.run_always("pdm", "install", "-G", "dev", "--no-self", external=True)
    session.run("pdm", "validate_code", external=True)
