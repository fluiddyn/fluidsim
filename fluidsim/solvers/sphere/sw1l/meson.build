python_sources = [
  '__init__.py',
  'solver.py',
  'state.py',
  'test_solver.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/solvers/sphere/sw1l'
)

run_command(['transonic', '--meson', '--backend', backend, 'solver.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
