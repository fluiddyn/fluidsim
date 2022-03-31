from util import submit_from_file, nb_nodes_from_N_1792


def type_fft_from_N(N):
    if N == 40:
        return "p3dfft"
    else:
        return "fftw1d"


submit_from_file(
    nh=1792,
    nh_small=1344,
    t_end=48.0,
    nb_nodes_from_N=nb_nodes_from_N_1792,
    type_fft_from_N=type_fft_from_N,
)
