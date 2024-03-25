python_sources = [
  '__init__.py',
  'dimensional.py',
  'test_dimensional.py',
  'forcing.py',
  'solver.py',
  'test_solver.py',
  'init_fields.py',
  'operators.py',
  'state.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/solvers/plate2d'
)

subdir('output')

run_command(['transonic', '--meson', '--backend', backend, 'operators.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
