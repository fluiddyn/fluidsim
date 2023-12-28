python_sources = [
  '__init__.py',
  'print_stdout.py',
  'spatial_means.py',
  'spatiotemporal_spectra.py',
  'spect_energy_budget.py',
  'spectra.py',
  'spectra_multidim.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/solvers/ns2d/output',
)

run_command(['transonic', '--meson', '--backend', backend, 'spatiotemporal_spectra.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
