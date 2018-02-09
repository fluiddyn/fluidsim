

# pythran export tendencies_nonlin_ns2dbouss(
#     float64[][], float64[][], float64[][], float64[][],
#     float64[][], float64[][])


def tendencies_nonlin_ns2dbouss(ux, uy, px_rot, py_rot, px_b, py_b):

    Frot = -ux*px_rot - uy*py_rot + px_b
    Fb = -ux*px_b - uy*py_b

    return Frot, Fb
