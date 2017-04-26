

def dealiasing_variable(ff_fft, where, nK0loc, nK1loc):
    for iK0 in range(nK0loc):
        for iK1 in range(nK1loc):
            if where[iK0, iK1]:
                ff_fft[iK0, iK1] = 0.


def dealiasing_setofvar(setofvar_fft, where, n0, n1):
    nk = setofvar_fft.shape[0]

    for i0 in range(n0):
        for i1 in range(n1):
            if where[i0, i1]:
                for ik in range(nk):
                    setofvar_fft[ik, i0, i1] = 0.
