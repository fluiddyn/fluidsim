

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

install:
	python setup.py install
