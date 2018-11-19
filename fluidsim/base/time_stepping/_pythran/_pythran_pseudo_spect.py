# pythran export rk2_step0(complex128[][][], complex128[][][], complex128[][][], float64[][], float)


def rk2_step0(state_spect_n12, state_spect, tendencies_n, diss2, dt):
    state_spect_n12[:] = (state_spect + dt / 2 * tendencies_n) * diss2


# pythran export rk2_step1(complex128[][][], complex128[][][], float64[][], float64[][], float)


def rk2_step1(state_spect, tendencies_n12, diss, diss2, dt):
    state_spect[:] = state_spect * diss + dt * diss2 * tendencies_n12


# pythran export rk4_step0(complex128[][][], complex128[][][], complex128[][][], complex128[][][], float64[][], float64[][], float)


def rk4_step0(
    state_spect,
    state_spect_tmp,
    tendencies_0,
    state_spect_np12_approx1,
    diss,
    diss2,
    dt,
):
    state_spect_tmp[:] = (state_spect + dt / 6 * tendencies_0) * diss
    state_spect_np12_approx1[:] = (state_spect + dt / 2 * tendencies_0) * diss2


# pythran export rk4_step1(complex128[][][], complex128[][][], complex128[][][], complex128[][][], float64[][], float)


def rk4_step1(
    state_spect,
    state_spect_tmp,
    state_spect_np12_approx2,
    tendencies_1,
    diss2,
    dt,
):
    state_spect_tmp[:] += dt / 3 * diss2 * tendencies_1
    state_spect_np12_approx2[:] = state_spect * diss2 + dt / 2 * tendencies_1


# pythran export rk4_step2(complex128[][][], complex128[][][], complex128[][][], complex128[][][], float64[][], float64[][], float)


def rk4_step2(
    state_spect,
    state_spect_tmp,
    state_spect_np1_approx,
    tendencies_2,
    diss,
    diss2,
    dt,
):
    state_spect_tmp[:] += dt / 3 * diss2 * tendencies_2
    state_spect_np1_approx[:] = state_spect * diss + dt * diss2 * tendencies_2


# pythran export arguments_blocks
arguments_blocks = {
    "rk2_step0": [
        "state_spect_n12",
        "state_spect",
        "tendencies_n",
        "diss2",
        "dt",
    ],
    "rk2_step1": ["state_spect", "tendencies_n12", "diss", "diss2", "dt"],
    "rk4_step0": [
        "state_spect",
        "state_spect_tmp",
        "tendencies_0",
        "state_spect_np12_approx1",
        "diss",
        "diss2",
        "dt",
    ],
    "rk4_step1": [
        "state_spect",
        "state_spect_tmp",
        "state_spect_np12_approx2",
        "tendencies_1",
        "diss2",
        "dt",
    ],
    "rk4_step2": [
        "state_spect",
        "state_spect_tmp",
        "state_spect_np1_approx",
        "tendencies_2",
        "diss",
        "diss2",
        "dt",
    ],
}
