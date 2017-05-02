

# pythran export dealiasing_setofvar(complex128[][][], uint8[][], int, int)

def dealiasing_setofvar(setofvar_fft, where, n0, n1):
    nk = setofvar_fft.shape[0]

    for i0 in range(n0):
        for i1 in range(n1):
            if where[i0, i1]:
                for ik in range(nk):
                    setofvar_fft[ik, i0, i1] = 0.
