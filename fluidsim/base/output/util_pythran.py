# pythran export strfunc_from_pdf(
#     float64[][], float64[][], float64[][], float, bool)

# pythran export strfunc_from_pdf(
#     int32[], float64[][], float64[][], float, bool)


import numpy as np


def strfunc_from_pdf(rxs, pdf, values, order, absolute=False):
    """Compute structure function of specified order from pdf for increments
    module.

    """
    S_order = np.empty(rxs.shape)
    if absolute:
        values = abs(values)
    for irx in range(rxs.size):
        deltainc = abs(values[irx, 1] - values[irx, 0])
        S_order[irx] = deltainc * np.sum(pdf[irx] * values[irx] ** order)

    return S_order
