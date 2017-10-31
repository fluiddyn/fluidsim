

develop:
	python setup.py develop

clean_so:
	find fluidsim -name "*.so" -delete

clean_pyc:
	find fluidsim -name "*.pyc" -delete
	find fluidsim -name "__pycache__" -type d | xargs rm -r

clean:
	rm -rf build

cleanall: clean clean_so

tests:
	python -m fluidsim.util.testing

tests_mpi:
	mpirun -np 2 python -m fluidsim.util.testing

_tests_coverage:
	mkdir -p .coverage
	coverage run -p -m fluidsim.util.testing
	mpirun -np 2 coverage run -p -m fluidsim.util.testing

_report_coverage:
	coverage combine
	coverage report
	coverage html
	coverage xml
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/.coverage/index.html"

coverage: _tests_coverage _report_coverage

lint:
	pylint -rn --rcfile=pylintrc --jobs=$(shell nproc) fluidsim

install:
	python setup.py install
