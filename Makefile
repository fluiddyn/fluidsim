
clean_so:
	find fluidsim -name "*.so" -delete
	find fluidsim -name "*.pyc" -delete

tests:
	python -m unittest discover

tests_mpi:
	mpirun -np 2 python -m unittest discover

develop:
	python setup.py develop

install:
	python setup.py install
