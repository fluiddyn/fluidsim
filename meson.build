project(
  'fluidsim',
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
py3 = py_mod.find_installation(pure: false)

pythran = find_program('pythran', native: true)
pythran_dep = declare_dependency(
  include_directories: incdir_pythran,
)

cpp_args_pythran = [
  '-DENABLE_PYTHON_MODULE',
  '-D__PYTHRAN__=3',
  '-DPYTHRAN_BLAS_NONE'
]

subdir('fluidsim')
