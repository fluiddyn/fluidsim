all:	clean_so clean_pyc clean develop

develop:
	python setup.py develop

clean_so:
	find fluidsim -name "*.so" -delete

clean_pyc:
	find fluidsim -name "*.pyc" -delete

clean:
	rm -rf build

tests:
	python -m unittest discover

tests_mpi:
	mpirun -np 2 python -m unittest discover

install:
	python setup.py install
