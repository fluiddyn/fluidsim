

develop:
	python setup.py develop

clean_so:
	find fluidsim -name "*.so" -delete

clean_pyc:
	find fluidsim -name "*.pyc" -delete

clean:
	rm -rf build

cleanall: clean clean_so

tests:
	python -m unittest discover

tests_mpi:
	mpirun -np 2 python -m unittest discover

tests_coverage:
	mkdir -p .coverage
	coverage run -m unittest discover
	# mpirun -np 2 coverage run -m unittest discover
	coverage report
	coverage html
	coverage xml
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/.coverage/index.html"

install:
	python setup.py install
