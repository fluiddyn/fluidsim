from dataclasses import dataclass

import matplotlib.pyplot as plt

import pymech

from fluidsim_core.hexa_files import SetOfPhysFieldFiles, HexaField


@dataclass
class MockEvent:
    inaxes: object
    xdata: float
    ydata: float


def test_setoffiles(false_output):

    output = false_output
    assert output.path_run.exists()

    hexa_data_loaded = pymech.readnek(
        output.path_run / f"session_00/{output.name_solver}0.f{1:05d}"
    )

    HexaField(hexa_data=hexa_data_loaded, key="vz")
    HexaField(hexa_data=hexa_data_loaded, key="pres", equation=None)

    HexaField(hexa_data=hexa_data_loaded, key="scalar", equation=None)
    HexaField(hexa_data=hexa_data_loaded, key="scalar 1", equation=None)

    set_of_files = SetOfPhysFieldFiles(output.path_run, output=output)
    hexa_z, time = set_of_files.get_field_to_plot(2, key="z", equation=None)

    set_of_files.get_key_field_to_plot("pressure")

    2.0 * hexa_z + hexa_z

    set_of_files.plot_hexa()
    set_of_files.plot_hexa(equation="y=0.75")
    set_of_files.plot_hexa(equation="x=0.5")

    set_of_files.plot_hexa(key="scalar 1", prefix="sts")
    set_of_files.plot_hexa(prefix="sts", time=0)

    ax = plt.gca()
    on_move = ax.figure._on_move_hexa
    event = MockEvent(ax, 0.5, 0.5)
    on_move(event)

    set_of_files.get_vector_for_plot(time=3)
    set_of_files.get_dataset_from_time(2.5)

    set_of_files.read_hexadata()
    set_of_files.read_hexadata(prefix="sts")
