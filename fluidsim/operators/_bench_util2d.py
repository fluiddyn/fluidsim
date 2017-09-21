
import perf

name_modules = (
    'native_openmp', 'native', 'openmp',
    'simd', 'simple')

runner = perf.Runner()


if __name__ == '__main__':

    for name in name_modules:
        name_module = 'util2d_pythran_bench_' + name

        setup = """

import {name_module} as mod

import numpy as np

shape = (100,) * 3
n0 = shape[1]
n1 = shape[2]

setofvars = np.ones(shape, dtype=np.complex128)
mask = np.ones(shape[1:], dtype=np.uint8)""".format(name_module=name_module)

        runner.timeit(
            name,
            'mod.dealiasing_setofvar(setofvars, mask, n0, n1)',
            setup=setup)
