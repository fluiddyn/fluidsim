project(
  'fluidsim',
  'cpp',
  license: 'CeCILL',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=release',
    'c_std=c99',
    'cpp_std=c++14',
  ],
)

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py = py_mod.find_installation('python3', pure: false)
py_dep = py.dependency()

backend = get_option('transonic-backend')
if backend == 'pythran'
  incdir_numpy = run_command(
    py,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)'''
    ],
    check: true
  ).stdout().strip()

  inc_np = include_directories(incdir_numpy)
  np_dep = declare_dependency(include_directories: inc_np)

  incdir_pythran = run_command(
      py,
      [
        '-c',
        '''import os
import pythran;
incdir = os.path.dirname(pythran.__file__)
try:
  incdir = os.path.relpath(incdir)
except Exception:
  pass
print(incdir)'''
      ],
      check: true
    ).stdout().strip()

  pythran = find_program('pythran', native: true)
  pythran_dep = declare_dependency(
    include_directories: incdir_pythran,
  )

  cpp_args_pythran = [
    '-DENABLE_PYTHON_MODULE',
    '-D__PYTHRAN__=3',
    '-DPYTHRAN_BLAS_NONE'
  ]
endif

subdir('fluidsim')
