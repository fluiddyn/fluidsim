"""Preprocessing for pseudo-spectral solvers (:mod:`fluiddyn.simul.base.preprocess.pseudo_spect`)
================================================================================================

.. currentmodule:: fluiddyn.simul.base.preprocess.base

Provides:

.. autoclass:: PreprocessBasePseudoSpectral
   :members:
   :private-members:


"""
import numpy as np
from .base import PreprocessBase

class PreprocessPseudoSpectral(PreprocessBase):
    _tag = 'pseudo_spectral'
    
    def __call__(self):
        if self.params.enable:
            self.normalize_init_fields()
            if self.sim.params.FORCING:
                self.set_forcing_rate()
            self.set_viscosity()            
            self.sim.output.save_info_solver_params_xml(replace=True)
    
    def normalize_init_fields(self):
        """
        A non-dimensionalization procedure for the initialized fields.
        """
        state = self.sim.state
        scale = self.params.init_field_scale
        
        if scale == 'energy':
            try:
                Ek, = self.output.compute_energies()
            except:
                Ek = self.output.compute_energy()
            u0 = np.sqrt(Ek)            # TODO: Why not sqrt(2.*Ek)
            ux_fft = state('ux_fft')
            uy_fft = state('uy_fft')
            ux_fft = ux_fft / u0
            uy_fft = uy_fft / u0
            try:
                self.sim.state.init_from_uxuyfft(ux_fft, uy_fft)
            except AttributeError:
                rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
                self.sim.state.init_statefft_from(rot_fft=rot_fft) 
            
    def set_viscosity(self):
        """
        Based on
         - the initial total enstrophy, \Omega_0, or 
         - the initial energy
         - the forcing rate, \epsilon
        the viscosity scale or Reynolds number is set.

        Parameters
        ----------
        params.preprocess.viscosity_type : string
            Type/Order of viscosity desired
        params.preprocess.viscosity_scale : string
            Mean quantity to be scaled against
        params.preprocess.viscosity_const : float
            Calibration constant to set dissipative wave number

        Note
        ----
        Algorithm: Sets viscosity variable nu and reinitializes f_d array for timestepping

        """
        params = self.params
        viscosity_type = params.viscosity_type
        viscosity_scale = params.viscosity_scale
        C = params.viscosity_const
        
        delta_x = self.oper.deltax
        k_max = np.pi / delta_x * self.sim.params.oper.coef_dealiasing # Smallest resolved scale
        length_scale = C * np.pi / k_max # OR np.pi / k_d, the dissipative wave number
    
        if viscosity_scale == 'enstrophy':                     
            omega_0 = self.output.compute_enstrophy()
            eta = omega_0 ** 1.5                     # Enstrophy dissipation rate
            time_scale = eta ** (-1. / 3)
        elif viscosity_scale == 'enstrophy_forcing':             
            omega_0 = self.output.compute_enstrophy()
            eta = omega_0 ** 1.5
            t1 = eta ** (-1. / 3)
            epsilon = self.sim.params.forcing.forcing_rate    # Energy dissipation rate
            t2 = epsilon ** (-1./3) * length_scale ** (2./3)
            time_scale = min(t1, t2)
        elif viscosity_scale == 'energy_enstrophy':            
            energy_0 = self.output.compute_energy()
            omega_0 = self.output.compute_enstrophy()
            epsilon = energy_0 * (omega_0 ** 0.5)
            time_scale =  epsilon ** (-1./3) * length_scale ** (2./3)
        elif viscosity_scale == 'forcing':
            epsilon = self.sim.params.forcing.forcing_rate
            time_scale =  epsilon ** (-1./3) * length_scale ** (2./3)
        else:
            raise ValueError('Unknown viscosity scale: %s'%viscosity_scale)
        
        self.sim.params.nu_2 = 0
        self.sim.params.nu_4 = 0
        self.sim.params.nu_8 = 0
        self.sim.params.nu_m4 = 0
        
        if viscosity_type == 'laplacian':
            self.sim.params.nu_2 = length_scale ** 2. / time_scale
        elif viscosity_type == 'hyper4':
            self.sim.params.nu_4 = length_scale ** 4. / time_scale
        elif viscosity_type == 'hyper8':
            self.sim.params.nu_8 = length_scale ** 8. / time_scale
        elif viscosity_type == 'hypo':
            self.sim.params.nu_m4 = length_scale ** (-4.) / time_scale
        else:
            raise ValueError('Unknown viscosity type: %s'%viscosity_type)
        
        self.sim.time_stepping.__init__(self.sim)

    def set_forcing_rate(self):
        """
        Based on
         - the initial total enstrophy, \Omega_0 
        the forcing rate is set.

        Parameters
        ----------
        params.preprocess.forcing_const : float
            Non-dimensional ratio of forcing_scale to forcing_rate
        params.preprocess.forcing_scale : string
            Mean quantity to be scaled against
        
        .. TODO : Trivial error in computing forcing rate
        """
        params = self.params
        
        forcing_scale = params.forcing_scale
        C = params.forcing_const
        delta_x = self.oper.deltax
        k_f = self.oper.deltakx * (self.sim.params.forcing.nkmax_forcing + 
                                   self.sim.params.forcing.nkmin_forcing) / 2 # Forcing wavenumber
        
        if forcing_scale == 'enstrophy':                     
            omega_0 = self.output.compute_enstrophy()
            self.sim.params.forcing.forcing_rate = (omega_0 / C) ** 1.5 / k_f ** 2
        else:
            raise ValueError('Unknown forcing scale: %s'% scale)

        self.sim.forcing.__init__(self.sim)
