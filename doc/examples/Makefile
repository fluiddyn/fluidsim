tests:
	rm -f .coverage
	pytest test_run_examples.py -v --cov --cov-config=.coveragerc $(PYTEST_ARGS) --durations=10
	coverage html
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/htmlcov/index.html"

tests_mpi:
	mpirun -np 2 pytest test_run_examples.py -v $(PYTEST_ARGS) --durations=10

install_requirements_tests:
	pip install -r requirements_tests.txt
