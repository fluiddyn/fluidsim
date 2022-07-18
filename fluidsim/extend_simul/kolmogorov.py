"""Kolmogorov flow (:mod:`fluidsim.base.forcing.kolmogorov`)
============================================================

.. autoclass:: KolmogorovFlow
   :members:
   :private-members:

.. autoclass:: KolmogorovFlowNormalized
   :members:
   :private-members:

"""

import numpy as np

from fluidsim_core.extend_simul import SimulExtender

from fluidsim.base.forcing.specific import (
    SpecificForcingPseudoSpectralSimple,
    NormalizedForcing,
)


class _KolmogorovFlowBase(SimulExtender):
    _module_name = "fluidsim.extend_simul.kolmogorov"
    tag = "kolmogorov_flow"

    @classmethod
    def get_modif_info_solver(cls):
        def modif_info_solver(info_solver):
            info_solver.classes.Forcing.classes._set_child(
                cls.tag,
                attribs={
                    "module_name": cls._module_name,
                    "class_name": cls.__name__,
                },
            )

        return modif_info_solver

    @classmethod
    def complete_params_with_default(cls, params):
        params.forcing.available_types.append(cls.tag)
        if not hasattr(params.forcing, "kolmo"):
            params.forcing._set_child(
                "kolmo",
                attribs={"ik": 3, "amplitude": None, "letter_gradient": "z"},
            )


class KolmogorovFlow(_KolmogorovFlowBase, SpecificForcingPseudoSpectralSimple):
    """Kolmogorov flow forcing

    Examples
    --------

    .. code-block:: python

        from fluidsim.solvers.ns3d.solver import Simul as SimulNotExtended

        from fluidsim.extend_simul import extend_simul_class
        from fluidsim.extend_simul.kolmogorov import KolmogorovFlow

        Simul = extend_simul_class(SimulNotExtended, KolmogorovFlow)

    """

    def __init__(self, sim):
        super().__init__(sim)
        params = sim.params

        ik = params.forcing.kolmo.ik
        amplitude = params.forcing.kolmo.amplitude
        if amplitude is None:
            amplitude = 1.0

        letter_gradient = params.forcing.kolmo.letter_gradient

        key_forced = params.forcing.key_forced
        if key_forced is None:
            key_forced = "vx"

        field = self.fstate.state_phys.get_var(key_forced)

        if len(sim.oper.axes) == 3:
            coords = sim.oper.get_XYZ_loc()
            lengths = [params.oper.Lx, params.oper.Ly, params.oper.Lz]
            letters = "xyz"
        else:
            raise NotImplementedError

        if letter_gradient not in letters:
            raise ValueError

        index = letters.index(letter_gradient)
        variable = coords[index]
        length = lengths[index]

        field[:] = amplitude * np.sin(2 * np.pi * ik / length * variable)
        self.fstate.statespect_from_statephys()

    def compute(self):
        # nothing to do here
        pass


class KolmogorovFlowNormalized(_KolmogorovFlowBase, NormalizedForcing):
    tag = "kolmogorov_flow_normalized"
