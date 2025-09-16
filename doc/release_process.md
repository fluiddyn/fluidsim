# Release process

- [ ] Extended tests in doc/examples
  (<https://foss.heptapod.net/fluiddyn/fluidsim/-/pipeline_schedules>)

- [ ] Check builds of the "official" articles in
  <https://foss.heptapod.net/fluiddyn/fluiddyn_papers>

- [ ] Topic/MR for the release candidate

  - [ ] Change versions in `pyproject.toml` AND `lib/pyproject.toml` (`0.6.1rc0`)

  - [ ] Update changelog in `CHANGES.md`

    - Take into account `doc/newsfragments` + remove the fragments

    - Visit
      <https://foss.heptapod.net/fluiddyn/fluidsim/-/compare/0.6.0...branch%2Fdefault>

    - Study `hg log -G -r "tag(0.6.0):tip"`

- [ ] Once merged, update, tag and push with `nox -s add-tag-for-release`.

- [ ] PR on <https://github.com/conda-forge/fluidsim-core-feedstock> (rc channel)

  In `recipe/conda_build_config.yaml` (see
  <https://conda-forge.org/docs/maintainer/knowledge_base.html#creating-a-pre-release-build>):

  ```yaml
  channel_targets:
    - conda-forge fluidsim-core_rc
  ```

  Check with `conda search fluidsim-core -c conda-forge/label/fluidsim-core_rc`

- [ ] PR on <https://github.com/conda-forge/fluidsim-feedstock> (rc channel)

  In `recipe/conda_build_config.yaml`:

  ```yaml
  channel_sources:
    - conda-forge/label/fluidsim-core_rc,conda-forge,defaults

  channel_targets:
    - conda-forge fluidsim_rc
  ```

  Check with `conda search fluidsim -c conda-forge/label/fluidsim_rc`

- [ ] Check the rc (with conda and doc/examples)

  Create new environment with

  ```bash
  mamba create -n env_fluidsim_rc \
    -c conda-forge/label/fluidsim-core_rc -c conda-forge/label/fluidsim_rc \
    fluidsim "fluidfft[build=mpi*]" "h5py[build=mpi*]"
  ```

- [ ] Communicate to the community...

- [ ] Topic/MR for the release of the stable version (delete "rc0" in the
  `pyproject.toml` files)

- [ ] New tag with `nox -s add-tag-for-release`

- [ ] PR on <https://github.com/conda-forge/fluidsim-core-feedstock>

- [ ] PR on <https://github.com/conda-forge/fluidsim-feedstock>

- [ ] Communicate to the community...
