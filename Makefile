
clean:
	rm -rf build
	find fluidsim -name "*.so" -delete
	find fluidsim -name "*.pyc" -delete

tests:
	python -m unittest discover

tests_mpi:
	mpirun -np 2 python -m unittest discover

tests_slurm:
	./scripts/tests_slurm.sh

develop:
	CC="cc"   \
	LDSHARED="cc -shared" \
	python setup.py develop
	#CFLAGS="-I/path/to/include"  \
	LDFLAGS="-L/path/to/lib"    \

install:
	python setup.py install
