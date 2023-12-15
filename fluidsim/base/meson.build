python_sources = [
  '__init__.py',
  'init_fields.py',
  'setofvariables.py',
  'state.py',
  'params.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidsim/base'
)

subdir('forcing')
subdir('output')
subdir('preprocess')
subdir('solvers')
subdir('sphericalharmo')
subdir('time_stepping')
subdir('turb_model')
subdir('test')
