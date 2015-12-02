

develop:
	python setup.py develop

clean_so:
	find fluidsim -name "*.so" -delete

tests:
	python -m unittest discover

tests_mpi:
	mpirun -np 2 python -m unittest discover
