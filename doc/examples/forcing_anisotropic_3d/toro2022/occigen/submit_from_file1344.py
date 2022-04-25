from util import submit_from_file, nb_nodes_from_N_1344


def type_fft_from_N(N):
    if N > 10:
        return "p3dfft"
    else:
        return "fftw1d"


submit_from_file(
    nh=1344,
    nh_small=896,
    t_end=44.0,
    nb_nodes_from_N=nb_nodes_from_N_1344,
    type_fft_from_N=type_fft_from_N,
)
