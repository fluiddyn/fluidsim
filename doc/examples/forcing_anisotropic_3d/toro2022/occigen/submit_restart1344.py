from util import submit_restart

if __name__ == "__main__":

    def get_nb_nodes(N):
        if N == 20:
            return 4
        else:
            return 8

    submit_restart(nh=1344, t_end=44.0, nb_nodes=get_nb_nodes)
