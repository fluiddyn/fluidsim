SHELL := bash
# Second tag after tip is usually the latest release
RELEASE=$(shell hg tags -T "{node|short}\n" | sed -n 2p)
MPI_NUM_PROCS ?= 2

.PHONY: black clean clean_pyc clean_so cleantransonic coverage_short develop develop_lib develop_user dist lint _report_coverage shortlog tests _tests_coverage tests_mpi

develop: develop_lib
	pip install -v -e .[dev] | grep -v link

develop_lib:
	cd lib && pip install -e .

develop_user:
	pip install -v -e .[dev] --user | grep -v link

develop_no-build-isolation: develop_lib
	pip install -e .[dev] --no-build-isolation

dist:
	cd lib && python setup.py sdist bdist_wheel
	python setup.py sdist
	mv -f lib/dist/* dist/

clean_so:
	find fluidsim -name "*.so" -delete

clean_pyc:
	find fluidsim -name "*.pyc" -delete
	find fluidsim -name "__pycache__" -type d | xargs rm -rf

cleantransonic:
	find fluidsim -type d -name __pythran__ | xargs rm -rf
	find fluidsim -type d -name __python__ | xargs rm -rf
	find fluidsim -type d -name __numba__ | xargs rm -rf
	find fluidsim -type d -name __cython__ | xargs rm -rf

clean:
	rm -rf build lib/build lib/dist

cleanall: clean clean_so cleantransonic

shortlog:
	@hg log -M -r$(RELEASE): --template '- {desc|firstline} (:rev:`{node|short}`)\n'

black:
	black -l 82 fluidsim scripts bench doc lib --exclude "/(__pythran__|doc/_build|\.ipynb_checkpoints/*)/"

tests:
	pytest -v lib
	fluidsim-test -v

tests_mpi:
	mpirun -np 2 fluidsim-test -v --exitfirst

define _test_mpi_fft_lib
	FLUIDSIM_TYPE_FFT=$(1) TRANSONIC_NO_REPLACE=1 mpirun -np $(2) --oversubscribe \
	  coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py
endef

_tests_coverage:
	mkdir -p .coverage
	coverage run -p -m pytest -v -s lib
	$(call _test_mpi_fft_lib,fft3d.mpi_with_fftwmpi3d,2)
	$(call _test_mpi_fft_lib,fft3d.mpi_with_fftw1d,2)
	# tests with p3dfft cannot be run together...
	FLUIDSIM_TYPE_FFT=fft3d.mpi_with_p3dfft TRANSONIC_NO_REPLACE=1 mpirun -np 2 --oversubscribe \
	  coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py -k "not TestCoarse"
	FLUIDSIM_TYPE_FFT=fft3d.mpi_with_p3dfft TRANSONIC_NO_REPLACE=1 mpirun -np 2 --oversubscribe \
	  coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py::TestCoarse
	FLUIDSIM_TYPE_FFT=fft3d.mpi_with_p3dfft TRANSONIC_NO_REPLACE=1 mpirun -np 4 --oversubscribe \
	  coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py::TestCoarse
	$(call _test_mpi_fft_lib,fft3d.mpi_with_pfft,2)
	$(call _test_mpi_fft_lib,fft3d.mpi_with_pfft,4)
	coverage run -p -m fluidsim.util.testing -v
	TRANSONIC_NO_REPLACE=1 coverage run -p -m fluidsim.util.testing -v
	TRANSONIC_NO_REPLACE=1 mpirun -np 2 --oversubscribe coverage run -p -m fluidsim.util.testing -v --exitfirst

_report_coverage:
	coverage combine
	coverage report
	coverage html
	coverage xml
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/.coverage/index.html"

coverage: _tests_coverage _report_coverage


lint:
	pylint -rn --rcfile=pylintrc --jobs=$(shell nproc) fluidsim --exit-zero

define _init_coverage
	rm -rf .coverage
	mkdir -p .coverage
endef

define _end_coverage
	coverage html
	@echo "Code coverage analysis complete. View detailed report:"
	@echo "file://${PWD}/.coverage/index.html"
endef

define _end_coverage_combine
	coverage combine
	$(call _end_coverage)
endef

coverage_short:
	$(call _init_coverage)
	coverage run -p -m pytest -v -s lib
	TRANSONIC_NO_REPLACE=1 coverage run -p -m fluidsim.util.testing -v -x
	make _report_coverage

pytest_cov_html:
	$(call _init_coverage)
	TRANSONIC_NO_REPLACE=1 pytest -v --cov --cov-config=setup.cfg $(PYTEST_ARGS) --durations=10 -x
	$(call _end_coverage)

pytest_cov_html_mpi:
	$(call _init_coverage)
	TRANSONIC_NO_REPLACE=1 mpirun -np $(MPI_NUM_PROCS) coverage run -p -m pytest -v --exitfirst $(PYTEST_ARGS)
	$(call _end_coverage_combine)

pytest_cov_html_full:
	$(call _init_coverage)
	TRANSONIC_NO_REPLACE=1 mpirun -np $(MPI_NUM_PROCS) coverage run -p -m pytest -v --exitfirst $(PYTEST_ARGS)
	TRANSONIC_NO_REPLACE=1 coverage run -p -m pytest -v $(PYTEST_ARGS) --durations=10
	$(call _end_coverage_combine)

define _pytest_mpi_operators3d
	$(call _init_coverage)
	FLUIDSIM_TYPE_FFT=$(1) TRANSONIC_NO_REPLACE=1 mpirun -np $(MPI_NUM_PROCS) coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py -p no:warnings
	$(call _end_coverage_combine)
endef

pytest_mpi_with_fftwmpi3d:
	$(call _pytest_mpi_operators3d,fft3d.mpi_with_fftwmpi3d)

pytest_mpi_with_fftw1d:
	$(call _pytest_mpi_operators3d,fft3d.mpi_with_fftw1d)

pytest_mpi_with_pfft:
	$(call _pytest_mpi_operators3d,fft3d.mpi_with_pfft)

pytest_mpi_with_p3dfft:
	$(call _init_coverage)
	FLUIDSIM_TYPE_FFT=fft3d.mpi_with_p3dfft TRANSONIC_NO_REPLACE=1 mpirun -np $(MPI_NUM_PROCS) \
	  coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py::TestCoarse
	FLUIDSIM_TYPE_FFT=fft3d.mpi_with_p3dfft TRANSONIC_NO_REPLACE=1 mpirun -np $(MPI_NUM_PROCS) \
	  coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py -k "not TestCoarse"
	$(call _end_coverage_combine)
