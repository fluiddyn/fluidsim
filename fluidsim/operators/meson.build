python_sources = [
  '__init__.py',
  'base.py',
  'operators0d.py',
  'operators1d.py',
  'operators2d.py',
  'operators3d.py',
  'op_finitediff1d.py',
  'op_finitediff2d.py',
  'sphericalharmo.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/operators'
)

subdir('test')

run_command(['transonic', '--meson', '--backend', backend, 'operators2d.py', 'operators3d.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
