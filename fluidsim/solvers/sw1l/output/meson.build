python_sources = [
  '__init__.py',
  'base.py',
  'increments.py',
  'normal_mode.py',
  '_old_spatial_means.py',
  '_old_spect_energy_budg.py',
  'print_stdout.py',
  'spatial_means.py',
  'spect_energy_budget.py',
  'spectra.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/solvers/sw1l/output'
)

run_command(['transonic', '--meson', '--backend', backend, 'base.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
