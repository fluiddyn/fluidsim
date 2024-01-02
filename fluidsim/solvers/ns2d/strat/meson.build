python_sources = [
  '__init__.py',
  'forcing.py',
  'init_fields.py',
  'solver.py',
  'state.py',
  'test_solver.py',
  'time_stepping.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/solvers/ns2d/strat',
)

subdir('output')

run_command(['transonic', '--meson', '--backend', backend, 'solver.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
