python_sources = [
  '__init__.py',
  'base.py',
  'finite_diff.py',
  'pseudo_spect.py',
  'simple.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/base/time_stepping'
)

run_command(['transonic', '--meson', '--backend', backend, 'pseudo_spect.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
