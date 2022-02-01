from math import pi
import sys

from fluiddyn.util import mpi

from fluidfft.fft3d.operators import OperatorsPseudoSpectral3D

arg = sys.argv[-1]

N = 64
L = 2 * pi

if arg.startswith("fft3d"):
    fft = arg
else:
    fft = None

print(f"fft = {fft}")

oper = OperatorsPseudoSpectral3D(N, N, N, L, L, L, fft=fft)

mpi.printby0(f"short name lib: {oper.oper_fft.get_short_name()}")

arr = oper.create_arrayX()
arr_fft = oper.create_arrayK()
oper.fft_as_arg(arr, arr_fft)

mpi.print_sorted(arr.shape)

mpi.print_sorted(arr_fft.shape)
