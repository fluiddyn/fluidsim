from util import submit_from_file, nb_nodes_from_N_2240


def type_fft_from_N(N):
    return "p3dfft"


submit_from_file(
    nh=2240,
    nh_small=1792,
    t_end=50.0,
    nb_nodes_from_N=nb_nodes_from_N_2240,
    type_fft_from_N=type_fft_from_N,
)
