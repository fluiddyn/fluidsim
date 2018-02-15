

# pythran export compute_Frot(
#     float64[][], float64[][], float64[][], float64[][], float)


def compute_Frot(ux, uy, px_rot, py_rot, beta=0):
    if beta == 0:
        return -ux*px_rot - uy*py_rot
    else:
        return -ux*px_rot - uy*(py_rot + beta)
