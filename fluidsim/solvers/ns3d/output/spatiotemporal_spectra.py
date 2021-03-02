"""
Spatiotemporal Spectra (:mod:`fluidsim.solvers.ns3d.output.spatiotemporal_spectra`)
===================================================================================

Provides:

.. autoclass:: SpatioTemporalSpectraNS3D
   :members:
   :private-members:

"""

from fluidsim.base.output.spatiotemporal_spectra import SpatioTemporalSpectra


class SpatioTemporalSpectraNS3D(SpatioTemporalSpectra):
    def plot(self):
        raise NotImplementedError
