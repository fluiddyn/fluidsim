"""State class for the SW1L.betaplane solver
(:mod:`fluidsim.solvers.sw1l.betaplane.state`)
===================================================


Provides:

.. autoclass:: StateSW1LBetaPlane
   :members:
   :private-members:

"""
from warnings import warn
import numpy as np

# from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.base.state import StatePseudoSpectral as StateBase

from fluiddyn.util import mpi

# from ..state import StateSW1L as StateBase


class StateSW1LBetaPlane(StateBase):
    """
    The class :class:`StateSW1LBetaPlane` contains the variables corresponding
    to the state and handles the access to other fields for the solver
    ``sw1l.betaplane``.
    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": ["rot_fft", "div_fft", "eta_fft"],
                "keys_state_phys": ["ux", "uy", "eta"],
                "keys_computable": [],
                "keys_phys_needed": ["ux", "uy", "eta", "rot", "div"],
                "keys_linear_eigenmodes": [],
            }
        )

    def __init__(self, sim, oper=None):

        super(StateSW1LBetaPlane, self).__init__(sim, oper)
        # TODO: define instead of field_tmp
        # self.state_jacobian = SetOfVariables(
        #     keys=["ux", "uy", "px_rot", "py_rot", "px_div", "py_div", "eta"]
        #)
        self.field_tmp0 = np.empty_like(self.state_phys[0])
        self.field_tmp1 = np.empty_like(self.state_phys[0])
        self.field_tmp2 = np.empty_like(self.state_phys[0])
        self.field_tmp3 = np.empty_like(self.state_phys[0])
        self.field_tmp4 = np.empty_like(self.state_phys[0])
        self.field_tmp5 = np.empty_like(self.state_phys[0])
        self.field_tmp6 = np.empty_like(self.state_phys[0])


    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        it = self.sim.time_stepping.it

        if key in self.vars_computed and it == self.it_computed[key]:
            return self.vars_computed[key]

        if key == "Jx":
            ux = self.state_phys.get_var("ux")
            eta = self.state_phys.get_var("eta")
            h = 1 + eta
            result = h * ux
        elif key == "Jy":
            uy = self.state_phys.get_var("uy")
            eta = self.state_phys.get_var("eta")
            h = 1 + eta
            result = h * uy
        elif key == "Jx_fft":
            Jx = self.compute("Jx")
            result = self.oper.fft2(Jx)
        elif key == "Jy_fft":
            Jy = self.compute("Jy")
            result = self.oper.fft2(Jy)
        elif key == "rot":
            rot_fft = self.state_spect.get_var("rot_fft")
            result = self.oper.ifft2(rot_fft)
        elif key == "div":
            div_fft = self.state_spect.get_var("div_fft")
            result = self.oper.ifft2(div_fft)
        elif key == "q":
            rot = self.compute("rot")
            eta = self.state_phys.get_var("eta")
            uy = self.state_phys.get_var("uy")
            result = rot - self.params.f * eta + self.params.beta * uy
        elif key == "q_fft":
            rot_fft = self.state_spect.get_var("rot_fft")
            eta_fft = self.state_spect.get_var("eta_fft")
            uy_fft = self.compute("uy_fft")
            result = rot_fft - self.params.f * eta_fft + self.params.beta * uy_fft
        elif key == "ux_fft":
            rot_fft = self.state_spect.get_var("rot_fft")
            div_fft = self.state_spect.get_var("div_fft")
            urx_fft, ury_fft = self.oper.vecfft_from_rotfft(rot_fft)
            udx_fft, udy_fft = self.oper.vecfft_from_divfft(div_fft)
            ux_fft = urx_fft + udx_fft
            if mpi.rank == 0:
                ux_fft[0, 0] = 0. + 0j  # TODO: Is it OK?
            result = ux_fft
            if SAVE_IN_DICT:
                key2 = "uy_fft"
                uy_fft = ury_fft + udy_fft
                if mpi.rank == 0:
                    uy_fft[0, 0] = 0. + 0j  # TODO: Is it OK?

                self.vars_computed[key2] = uy_fft
                self.it_computed[key2] = it

        elif key == "uy_fft":
            rot_fft = self.state_spect.get_var("rot_fft")
            div_fft = self.state_spect.get_var("div_fft")
            urx_fft, ury_fft = self.oper.vecfft_from_rotfft(rot_fft)
            udx_fft, udy_fft = self.oper.vecfft_from_divfft(div_fft)
            uy_fft = ury_fft + udy_fft
            if mpi.rank == 0:
                uy_fft[0, 0] = 0. + 0j  # TODO: Is it OK?
            result = uy_fft
            if SAVE_IN_DICT:
                key2 = "ux_fft"
                ux_fft = urx_fft + udx_fft
                if mpi.rank == 0:
                    ux_fft[0, 0] = 0. + 0j  # TODO: Is it OK?
                self.vars_computed[key2] = ux_fft
                self.it_computed[key2] = it

        else:
            warn("Using super methods for computing {}".format(key))
            #  result = super(StateSW1LBetaPlane, self).compute(
            #      key, SAVE_IN_DICT=SAVE_IN_DICT, RAISE_ERROR=RAISE_ERROR
            #  )
            result = super(StateSW1LBetaPlane, self).compute(key)
            SAVE_IN_DICT = False

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

        return result

    def statespect_from_statephys(self):
        """Compute the state in Fourier space."""
        ux = self.state_phys.get_var("ux")
        uy = self.state_phys.get_var("uy")
        eta = self.state_phys.get_var("eta")
        
        rot_fft = self.state_spect.get_var("rot_fft")
        div_fft = self.state_spect.get_var("div_fft")
        eta_fft = self.state_spect.get_var("eta_fft")

        self.oper.fft_as_arg(eta, eta_fft)
        ux_fft = self.oper.fft2(ux)
        uy_fft = self.oper.fft2(uy)

        rot_fft[:] = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        div_fft[:] = self.oper.divfft_from_vecfft(ux_fft, uy_fft)

    def statephys_from_statespect(self):
        """Compute the state in physical space."""
        rot_fft = self.state_spect.get_var("rot_fft")
        div_fft = self.state_spect.get_var("div_fft")
        eta_fft = self.state_spect.get_var("eta_fft")

        rot_fft = self.state_spect.get_var("rot_fft")
        div_fft = self.state_spect.get_var("div_fft")
        ux_fft, uy_fft = self.oper.vecfft_from_rotdivfft(rot_fft, div_fft)

        ux = self.state_phys.get_var("ux")
        uy = self.state_phys.get_var("uy")
        eta = self.state_phys.get_var("eta")

        self.oper.ifft_as_arg(ux_fft, ux)
        self.oper.ifft_as_arg(uy_fft, uy)
        self.oper.ifft_as_arg(eta_fft, eta)

    def init_from_etafft(self, eta_fft):
        r"""Initialize from :math:`\hat \eta` and set velocities to zero."""
        state_spect = self.state_spect
        state_spect.set_var("rot_fft", np.zeros_like(eta_fft))
        state_spect.set_var("div_fft", np.zeros_like(eta_fft))
        state_spect.set_var("eta_fft", eta_fft)

        self.oper.dealiasing(state_spect)
        self.statephys_from_statespect()

    def init_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft):
        """Self explanatory."""
        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        div_fft = self.oper.divfft_from_vecfft(ux_fft, uy_fft)
        state_spect = self.state_spect
        state_spect.set_var("rot_fft", rot_fft)
        state_spect.set_var("div_fft", div_fft)
        state_spect.set_var("eta_fft", eta_fft)

        self.oper.dealiasing(state_spect)
        self.statephys_from_statespect()

    def init_from_rotfft(self, rot_fft):
        """Initializes with rotational velocities computed from vorticity."""
        state_spect = self.state_spect
        state_spect.set_var("rot_fft", rot_fft)
        state_spect.set_var("div_fft", np.zeros_like(rot_fft))
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        ux = self.oper.ifft(ux_fft)
        uy = self.oper.ifft(uy_fft)
        rot = self.oper.ifft(rot_fft)
        state_spect.set_var(
            "eta_fft",
            self._etafft_no_div(ux, uy, rot)
        )

    def init_from_etafft(self, eta_fft):
        """Initializes with rotational velocities computed from vorticity."""
        state_spect = self.state_spect
        # state_spect.set_var("rot_fft", rot_fft)
        # state_spect.set_var("div_fft", div_fft)
        raise NotImplementedError
        state_spect.set_var("eta_fft", eta_fft)

    def init_from_qfft(self, q_fft):
        """Initialize from potential vorticity."""
        # TODO: check if the oper functions are fine with beta term
        rot_fft = self.oper.rotfft_from_qfft(q_fft)
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        eta_fft = self.oper.etafft_from_qfft(q_fft)
        self.init_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

    def init_from_afft(self, a_fft):
        """Initialize from ageostrophic variable."""
        ux_fft, uy_fft, eta_fft = self.oper.uxuyetafft_from_afft(a_fft)
        self.init_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

    def init_from_qafft(self, q_fft, a_fft):
        """Initialize from potential vorticity and ageostrophic variables."""
        rot_fft = self.oper.rotfft_from_qfft(q_fft)
        uxq_fft, uyq_fft = self.oper.vecfft_from_rotfft(rot_fft)
        etaq_fft = self.oper.etafft_from_qfft(q_fft)

        uxa_fft, uya_fft, etaa_fft = self.oper.uxuyetafft_from_afft(a_fft)
        self.init_from_uxuyetafft(
            uxq_fft + uxa_fft, uyq_fft + uxa_fft, etaq_fft + etaa_fft
        )

    def init_from_uxuyfft(self, ux_fft, uy_fft):
        """Initialize from velocities and adjust surface displacement by solving
        a Poisson equation, assuming no mean flow.

        """
        oper = self.oper
        ifft2 = oper.ifft2

        oper.projection_perp(ux_fft, uy_fft)
        oper.dealiasing(ux_fft, uy_fft)

        ux = ifft2(ux_fft)
        uy = ifft2(uy_fft)

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        rot = ifft2(rot_fft)

        eta_fft = self._etafft_no_div(ux, uy, rot)
        eta = ifft2(eta_fft)

        state_spect = self.state_spect
        state_spect.set_var("ux_fft", ux_fft)
        state_spect.set_var("uy_fft", uy_fft)
        state_spect.set_var("eta_fft", eta_fft)

        state_phys = self.state_phys
        state_phys.set_var("rot", rot)
        state_phys.set_var("ux", ux)
        state_phys.set_var("uy", uy)
        state_phys.set_var("eta", eta)

    def init_statespect_from(self, **kwargs):
        """Initializes *state_spect* using arrays provided as keyword
        arguments.

        """
        if len(kwargs) == 1:
            key_fft, value = list(kwargs.items())[0]
            try:
                init_from_keyfft = self.__getattribute__(
                    "init_from_" + key_fft.replace("_", "")
                )
                init_from_keyfft(value)
            except AttributeError:
                super(StateSW1LBetaPlane, self).init_statespect_from(**kwargs)
        elif len(kwargs) == 2:
            if "q_fft" in kwargs and "a_fft" in kwargs:
                self.init_from_qafft(**kwargs)
        else:
            super(StateSW1LBetaPlane, self).init_statespect_from(**kwargs)

    def _etafft_no_div(self, ux, uy, rot):
        raise NotImplementedError  # FIXME: Check Poisson equation with beta term
        oper = self.oper
        rot_abs = rot + self.params.f

        tempx_fft = -self.oper.fft2(rot_abs * uy)
        tempy_fft = self.oper.fft2(rot_abs * ux)

        uu2_fft = self.oper.fft2(ux ** 2 + uy ** 2)

        eta_fft = ((
                oper.invlaplacian_fft(
                        oper.divfft_from_vecfft(tempx_fft, tempy_fft), negative=True)
                - (uu2_fft / 2)
            ) /
            self.params.c2
        )
        if mpi.rank == 0:
            eta_fft[0, 0] = 0.
        self.oper.dealiasing(eta_fft)

        return eta_fft
