
# dont pythran export compute_Frot(
#     float64[][], float64[][], float64[][], float64[][], float)


# pythran export compute_Fdiv(
#     float64[][], float64[][], float64[][], float64[][], float)

#
#
#def compute_Frot(ux, uy, px_rot, py_rot, beta=0):
#    if beta == 0:
#        Frot = -ux * px_rot - uy * py_rot
#
#    else:
#        Frot = -ux * px_rot - uy * (py_rot + beta)
#
#    if f != 0:
#        Frot -= f * div
#    
#    return Frot


def compute_Fdiv(ux, uy, px_div, py_div, beta=0):
    if beta == 0:
        Fdiv = -ux * px_div - uy * py_div

    else:
        Fdiv = -ux * (px_div + beta) - uy * py_div

    #    if f != 0:
    #        Fdiv += f * rot
    
    return Fdiv