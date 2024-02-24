
PYTEST_ARGS?=test_run_examples.py

test:
	rm -f .coverage
	pytest $(PYTEST_ARGS) -v --cov --cov-config=.coveragerc --durations=10
	coverage html
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/htmlcov/index.html"

test_mpi:
	mpirun -np 2 pytest $(PYTEST_ARGS) -v --durations=10

install_requirements_tests:
	pip install -r requirements_tests.txt
