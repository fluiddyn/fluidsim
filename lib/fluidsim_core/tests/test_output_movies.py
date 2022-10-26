from fluidsim_core.output.phys_fields_snek5000 import PhysFields4Snek5000


def test_hexa_movies(false_output):

    phys_fields = PhysFields4Snek5000(false_output)
    phys_fields.animate(normalize_vectors=True, clim=(0, 1))
    phys_fields.movies.update_animation(1)
