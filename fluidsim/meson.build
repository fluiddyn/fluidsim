python_sources = [
  '__init__.py',
  '_version.py',
  'magic.py',
  'build_conf.txt',
]

run_command(
  'sh',
  '-c',
  'echo transonic_backend=' + backend + ' > ' + 'build_conf.txt',
  check: true,
)


py.install_sources(
  python_sources,
  subdir: 'fluidsim'
)

subdir('base')
subdir('solvers')
subdir('extend_simul')
subdir('operators')
subdir('util')
