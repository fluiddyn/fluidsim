from pathlib import Path

from fluidsim.util.scripts.test_restart import path_simul

from fluidsim.util.scripts.modif_resolution import main


def test_with_t_approx(path_simul):
    args = [path_simul, "2"]
    main(args, t_approx=1000)
    assert list(Path(path_simul).glob("State_phys_24x24/state_phys_t*"))


def test_with_path_file(path_simul):
    path_simul = Path(path_simul)
    path_last_time = str(sorted(path_simul.glob("state_phys*"))[-1])
    args = [path_last_time, "3/2"]
    main(args)
    assert list(path_simul.glob("State_phys_18x18/state_phys_t*"))
