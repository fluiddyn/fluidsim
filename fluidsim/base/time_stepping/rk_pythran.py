
# pythran export step0_RK2_pythran(
#     complex128[][][],
#     complex128[][][],
#     complex128[][][],
#     float64[][] or complex128[][],
#     float
# )

# pythran export step0_RK2_pythran(
#     complex128[][][][],
#     complex128[][][][],
#     complex128[][][][],
#     float64[][][] or complex128[][][],
#     float
# )


def step0_RK2_pythran(state_spect_n12, state_spect, tendencies_n, diss2, dt):
    state_spect_n12[:] = (state_spect + dt / 2 * tendencies_n) * diss2


# pythran export step1_RK2_pythran(
# complex128[][][],
# complex128[][][],
# float64[][] or complex128[][],
# float64[][] or complex128[][],
# float
# )

# pythran export step1_RK2_pythran(
# complex128[][][][],
# complex128[][][][],
# float64[][][] or complex128[][][],
# float64[][][] or complex128[][][],
# float
# )

def step1_RK2_pythran(state_spect, tendencies_n12, diss, diss2, dt):
    state_spect[:] = state_spect * diss + dt * diss2 * tendencies_n12
