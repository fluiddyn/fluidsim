python_sources = [
  '__init__.py',
  'frequency_modulation.py',
  'mini_oper_modif_resol.py',
  'phys_fields.py',
  'testing.py',
  'test_util.py',
  'test_phys_fields.py',
  'util.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/util'
)

subdir('console')
subdir('scripts')

run_command(['transonic', '--meson', '--backend', backend, 'mini_oper_modif_resol.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
