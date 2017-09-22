

# pythran export dealiasing_setofvar(complex128[][][], uint8[][], int, int)

def dealiasing_setofvar(setofvar_fft, where, n0, n1):
    nk = setofvar_fft.shape[0]

    for i0 in range(n0):
        for i1 in range(n1):
            if where[i0, i1]:
                for ik in range(nk):
                    setofvar_fft[ik, i0, i1] = 0.


# pythran export laplacian2_fft(complex128[][], float64[][])

def laplacian2_fft(a_fft, K4):
    return a_fft * K4


# pythran export invlaplacian2_fft(complex128[][], float64[][], int)

def invlaplacian2_fft(a_fft, K4_not0, rank):
    invlap2_afft = a_fft / K4_not0
    if rank == 0:
        invlap2_afft[0, 0] = 0.
    return invlap2_afft
