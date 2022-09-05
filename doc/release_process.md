# Release process

- [ ] Extended tests in doc/examples (`make tests` and `make tests_mpi`)

- [ ] Check builds of the "official" articles in <https://foss.heptapod.net/fluiddyn/fluiddyn_papers>

- [ ] Topic for the release candidate

  - [ ] Change version in `lib/fluidsim_core/_version.py` (`0.6.1rc0`)

  - [ ] Update changelog in `CHANGES.rst`

    - Take into account `doc/newsfragments` + remove the fragments

    - Visit <https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.6.0...branch%2Fdefault>

    - Study `hg log -G -r "tag(0.6.0):tip"`

- [ ] Tag `0.6.1rc0` in the repo

- [ ] Push the release candidate to PyPI (no wheel for fluidsim!)

  ```bash
  cd lib
  python setup.py sdist bdist_wheel
  twine upload dist/*
  cd ..
  python setup.py sdist
  twine upload dist/*
  ```

- [ ] PRs on <https://github.com/conda-forge/fluidsim-core-feedstock> and
  <https://github.com/conda-forge/fluidsim-feedstock> (special rc channel)

  In `recipe/conda_build_config.yaml`:

  ```yaml
  channel_sources:
    - conda-forge/label/fluidsim-core_rc,conda-forge,defaults

  channel_targets:
    - conda-forge fluidsim_rc
  ```

  - <https://conda-forge.org/docs/maintainer/knowledge_base.html#creating-a-pre-release-build>
  - <https://github.com/conda-forge/fluidsim-feedstock/pull/14/files#diff-ff61408cdc05bc9667deeadb55e4aaceb1371972076b6bf6934f9008920f2bd2>

- [ ] Check the rc (with conda and doc/examples)

  ```bash
  conda create -n env_fluidsim_rc -c conda-forge/label/fluidsim-core_rc -c conda-forge/label/fluidsim_rc fluidsim "fluidfft[build=mpi*]" "h5py[build=mpi*]"
  ```

- [ ] Communicate to the community...

- [ ] Topic for the release of the stable version (delete "rc0" in `lib/fluidsim_core/_version.py`)

- [ ] Tag `0.6.1` in the repo

- [ ] Push to PyPI (no wheel for fluidsim!)

- [ ] PRs on <https://github.com/conda-forge/fluidsim-core-feedstock> and
  <https://github.com/conda-forge/fluidsim-feedstock> (special rc channel)

- [ ] Communicate to the community...
