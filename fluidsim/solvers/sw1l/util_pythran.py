
# pythran export compute_Frot(
#     float64[][], float64[][], float64[][],
#     float or int)
def compute_Frot(rot, ux, uy, f):
    """Compute cross-product of absolute potential vorticity with velocity."""
    if f != 0:
        rot_abs = rot + f
    else:
        rot_abs = rot

    F1x = rot_abs * uy
    F1y = -rot_abs * ux

    return F1x, F1y
