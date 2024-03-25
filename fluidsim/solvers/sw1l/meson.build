python_sources = [
  '__init__.py',
  'forcing.py',
  'init_fields.py',
  'operators.py',
  'solver.py',
  'state.py',
  'test_operators.py',
  'test_solver.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/solvers/sw1l',
)

subdir('exactlin')
subdir('modified')
subdir('onlywaves')
subdir('output')

run_command(
  ['transonic', '--meson', '--backend', backend, 'operators.py', 'solver.py'],
  check: true
)

foreach be : backends
  subdir('__' + be + '__')
endforeach
