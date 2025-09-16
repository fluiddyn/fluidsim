SHELL := bash
# Second tag after tip is usually the latest release
RELEASE=$(shell hg tags -T "{node|short}\n" | sed -n 2p)
MPI_NUM_PROCS ?= 2

.PHONY: black black_check clean clean_pyc clean_so cleantransonic coverage_short develop dist lint _report_coverage shortlog tests _tests_coverage tests_mpi

develop: sync

sync:
	pdm sync

sync-clean:
	pdm sync --clean

sync-no-self:
	pdm sync --no-self

install_fluidfft_plugins:
	pdm run pip install fluidfft-fftw fluidfft-fftwmpi fluidfft-mpi_with_fftw

install_editable_perf:
	pdm sync --clean --no-self
	pip install -e . -v --no-build-isolation --no-deps -C setup-args=-Dnative=true

lock:
	pdm lock

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

format:
	@# much faster than `pdm run`
	.venv/bin/ruff format *.py fluidsim scripts bench doc lib

check-format:
	pdm check-format

lint:
	pdm lint

validate_code:
	pdm validate_code

tests:
	pytest -v lib
	fluidsim-test -v

tests_mpi:
	mpirun -np 2 fluidsim-test -v --exitfirst


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
	TRANSONIC_NO_REPLACE=1 pytest -v --cov --cov-config=pyproject.toml $(PYTEST_ARGS) --durations=10 -x
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
	FLUIDSIM_TYPE_FFT3D=$(1) TRANSONIC_NO_REPLACE=1 mpirun -np $(MPI_NUM_PROCS) coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py -p no:warnings
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
	FLUIDSIM_TYPE_FFT3D=fft3d.mpi_with_p3dfft TRANSONIC_NO_REPLACE=1 mpirun -np $(MPI_NUM_PROCS) \
	  coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py::TestCoarse
	FLUIDSIM_TYPE_FFT3D=fft3d.mpi_with_p3dfft TRANSONIC_NO_REPLACE=1 mpirun -np $(MPI_NUM_PROCS) \
	  coverage run -p -m pytest -v --exitfirst fluidsim/operators/test/test_operators3d.py -k "not TestCoarse"
	$(call _end_coverage_combine)
