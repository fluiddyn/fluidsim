python_sources = [
  '__init__.py',
  'correlations_freq.py',
  'print_stdout.py',
  'spatial_means.py',
  'spectra.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/solvers/plate2d/output'
)

run_command(['transonic', '--meson', '--backend', backend, 'correlations_freq.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
