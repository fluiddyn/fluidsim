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

- [ ] Push the release candidate to PyPI (no wheel!)

- [ ] PR on <https://github.com/conda-forge/fluidsim-feedstock> (special channel, see <https://foss.heptapod.net/fluiddyn/fluidsim/-/issues/97>)

- [ ] Check the rc (with conda and doc/examples)

- [ ] Communicate to the community...

- [ ] Topic for the release of the stable version (delete "rc0" in `lib/fluidsim_core/_version.py`)

- [ ] Tag `0.6.1` in the repo

- [ ] Push to PyPI (no wheel!)

- [ ] PR on <https://github.com/conda-forge/fluidsim-feedstock>

- [ ] Communicate to the community...
