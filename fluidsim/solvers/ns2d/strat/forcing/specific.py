"""Forcing (:mod:`fluidsim.solvers.ns2d.strat.forcing.specific`)
===============================================================

.. autoclass:: SpecificForcingPseudoSpectralAnisotrop
   :members:

"""

from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from math import radians

import numpy as np

from copy import deepcopy

from fluiddyn.util import mpi
from fluidsim.base.forcing.specific import SpecificForcing
from fluidsim.base.setofvariables import SetOfVariables


class SpecificForcingPseudoSpectralAnisotrop(SpecificForcing):
    tag = 'pseudo_spectral_anisotrop'
    _key_forced_default = 'rot_fft'

    def __init__(self, sim):

        super(SpecificForcingPseudoSpectralAnisotrop, self).__init__(sim)

        params = sim.params

        self.sum_wavenumbers = sim.oper.sum_wavenumbers
        self.fft2 = sim.oper.fft2
        self.ifft2 = sim.oper.ifft2

        self.forcing_fft = SetOfVariables(
            like=sim.state.state_fft, info='forcing_fft', value=0.)

        self.kmax_forcing = self.oper.deltakx*params.forcing.nkmax_forcing
        self.kmin_forcing = self.oper.deltakx*params.forcing.nkmin_forcing
        
        self.forcing_rate = params.forcing.forcing_rate

        if params.forcing.key_forced is not None:
            self.key_forced = params.forcing.key_forced
        else:
            self.key_forced = self._key_forced_default

        i = 0
        while 2*params.forcing.nkmax_forcing > 2**i:
            i += 1
        n = 2**i

        if mpi.rank == 0:
            params_coarse = deepcopy(params)
            params_coarse.oper.nx = n
            params_coarse.oper.ny = n
            params_coarse.oper.type_fft = 'sequential'
            params_coarse.oper.coef_dealiasing = 3.  # FIXME: Workaround for incorrect forcing

            self.oper_coarse = sim.oper.__class__(
                SEQUENTIAL=True,
                params=params_coarse,
                goal_to_print='coarse resolution for forcing')
            self.shapeK_loc_coarse = self.oper_coarse.shapeK_loc

            self.COND_NO_F = np.logical_or(
                self.oper_coarse.KK > self.kmax_forcing,
                self.oper_coarse.KK < self.kmin_forcing)

            self.nb_forced_modes = (self.COND_NO_F.size -
                                    np.array(self.COND_NO_F,
                                             dtype=np.int32).sum())
            self.ind_forcing = np.logical_not(
                self.COND_NO_F).flatten().nonzero()[0]

            self.fstate_coarse = sim.state.__class__(
                sim, oper=self.oper_coarse)

        else:
            self.shapeK_loc_coarse = None

        if mpi.nb_proc > 1:
            self.shapeK_loc_coarse = mpi.comm.bcast(
                self.shapeK_loc_coarse, root=0)

        # if params.forcing.type_forcing == 'WAVES':
        #     self.compute = self.compute_forcing_waves
        #     if mpi.rank == 0:
        #         eta_rms_max = 0.1
        #         self.eta_cond = eta_rms_max / np.sqrt(self.nb_forced_modes)
        #         print '    eta_cond =', self.eta_cond

    def compute(self):
        """compute the forcing from a coarse forcing."""

        a_fft = self.sim.state.state_fft.get_var(self.key_forced)
        a_fft = self.oper.coarse_seq_from_fft_loc(a_fft,
                                                  self.shapeK_loc_coarse)

        if mpi.rank == 0:
            Fa_fft = self.forcingc_raw_each_time(a_fft)
            self.fstate_coarse.init_statefft_from(**{self.key_forced: Fa_fft})

        self.put_forcingc_in_forcing()

    def put_forcingc_in_forcing(self):
        """Copy data from forcingc_fft into forcing_fft."""
        nKyc = self.shapeK_loc_coarse[0]
        nKxc = self.shapeK_loc_coarse[1]
        nb_keys = self.forcing_fft.nvar

        ar3Df = self.forcing_fft
        if mpi.rank == 0:
            # ar3Dfc = self.forcingc_fft
            ar3Dfc = self.fstate_coarse.state_fft

        if mpi.nb_proc > 1:
            nKy = self.oper.shapeK_seq[0]

            for ikey in range(nb_keys):
                if mpi.rank == 0:
                    fck_fft = ar3Dfc[ikey].transpose()

                for iKxc in range(nKxc):
                    kx = self.oper.deltakx*iKxc
                    rank_iKx, iKxloc, iKyloc = (
                        self.oper.where_is_wavenumber(kx, 0.))

                    if mpi.rank == 0:
                        fc1D = fck_fft[iKxc]

                    if rank_iKx != 0:
                        # message fc1D
                        if mpi.rank == rank_iKx:
                            fc1D = np.empty([nKyc], dtype=np.complex128)
                        if mpi.rank == 0 or mpi.rank == rank_iKx:
                            fc1D = np.ascontiguousarray(fc1D)
                        if mpi.rank == 0:
                            mpi.comm.Send([fc1D, mpi.MPI.COMPLEX],
                                          dest=rank_iKx, tag=iKxc)
                        elif mpi.rank == rank_iKx:
                            mpi.comm.Recv([fc1D, mpi.MPI.COMPLEX],
                                          source=0, tag=iKxc)
                    if mpi.rank == rank_iKx:
                        # copy
                        for iKyc in range(nKyc):
                            if iKyc <= nKyc/2.:
                                iKy = iKyc
                            else:
                                kynodim = iKyc - nKyc
                                iKy = kynodim + nKy
                            ar3Df[ikey, iKxloc, iKy] = fc1D[iKyc]

        else:
            nKy = self.oper.shapeK_seq[0]

            for ikey in range(nb_keys):
                for iKyc in range(nKyc):
                    if iKyc <= nKyc/2.:
                        iKy = iKyc
                    else:
                        kynodim = iKyc - nKyc
                        iKy = kynodim + nKy
                    for iKxc in range(nKxc):
                        ar3Df[ikey, iKy, iKxc] = ar3Dfc[ikey, iKyc, iKxc]

    def verify_injection_rate(self):
        """Verify injection rate."""
        Fa_fft = self.forcing_fft.get_var(self.key_forced)
        a_fft = self.sim.state.state_fft.get_var(self.key_forced)

        PZ_forcing1 = abs(Fa_fft)**2/2*self.sim.time_stepping.deltat
        PZ_forcing2 = np.real(
            Fa_fft.conj()*a_fft +
            Fa_fft*a_fft.conj()) / 2.
        PZ_forcing1 = self.oper.sum_wavenumbers(PZ_forcing1)
        PZ_forcing2 = self.oper.sum_wavenumbers(PZ_forcing2)
        if mpi.rank == 0:
            print('PZ_f = {0:9.4e} ; PZ_f2 = {1:9.4e};'.format(
                PZ_forcing1 + PZ_forcing2,
                PZ_forcing2))



# class TimeCorrelatedRandomPseudoSpectralAnisotrop(
#         TimeCorrelatedRandomPseudoSpectral):
#     tag = 'tcrandom_anisotropic'

# Define functions for anisotropic forcing!!!
