import numpy as np
import astropy.units as astropy_units

#from astromodels import Gaussian

from astromodels.functions.function import (
    Function1D,
    FunctionMeta,
    ModelAssertionViolation,
)


class Positronium_Spectrum(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        Positronium spectrum, define by the ortho-Ps (3-photon decay) and para-Ps (2-photon decay = line) component.
        Physical bounds are included so that the ortho-Ps spectrum is at most 4.5 times the para-Ps spectrum (maximum allowed by quantum statistics),
        given by the ratio as a function of the Positronium fraction f_Ps. The flux F is chosen as the line flux as a normalisation of the total spectrum.
    latex : $ F \times \left[ L(E;\mu,\sigma) + r(f_{\rm Ps}) O(E;\mu) \right] $
    parameters :
        Flux :
            desc : Normalization of flux (chosen to be line flux)
            initial value : 0.001
            is_normalization : True
            min : 0
            max : 1e3
            delta : 0.1
        mu : 
            desc : centroid of the 511 keV line
            initial value : 511.0
            min : 490
            max : 530
            fix : True
        sigma :
            desc : width of the 511 keV line
            initial value : 1.4
            min : 0.81
            max : 20.0
            fix : True
        f_Ps :
            desc : Positronium fraction
            initial value : 1
            min : 0
            max : 1

    """

    def _set_units(self, x_unit, y_unit):
        # line flux units
        self.Flux.unit = 1/(astropy_units.cm**2 * astropy_units.s)

        # centroid unit
        self.mu.unit = astropy_units.keV

        # width unit
        self.sigma.unit = astropy_units.keV

        # positronium fraciton is dimensionless
        self.f_Ps.unit = astropy_units.dimensionless_unscaled

    # noinspection PyPep8Naming
    def evaluate(self, x, Flux, mu, sigma, f_Ps):

        """
        # check is the function is being called with units
        if isinstance(Flux, astropy_units.Quantity):

            # get the values
            Flux_ = Flux.value
            x_    = x.value
            f_Ps_ = f_Ps.value
            # fixed values
            sigma = 1.4
            mu    = 511.0

            unit_ = self.y_unit
            
        else:
            
            # we do not need to do anything here
            Flux_  = Flux
            f_Ps_  = f_Ps
            sigma  = 1.4*astropy_units.keV
            mu     = 511.0*astropy_units.keV
            x_     = x

            unit_ = 1.0
        """
         
        # evaluate components
        # 511 keV line
        Line = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*((x-mu)/sigma)**2)

        # ortho-Positronium
        m = mu

        t1 = (x*(m-x)) / (2*m-x)**2
        t2 = (2*m*(m-x)**2)/(2*m-x)**3
        t3 = (m-x)/m
        t4 = (2*m-x)/x
        t5 = 2*m*(m-x)/x**2
        t6 = (m-x)/m

        oPs_norm = 1*m*(np.pi**2 - 9)
        
        oPs = 1 / oPs_norm * 2 * (t1 - t2*np.log(t3) + t4 + t5*np.log(t6))
        oPs[~np.isfinite(oPs)] = 0.

        # ratio oPs/Line  from positonium fraction
        ratio = (9 * f_Ps) / (8 - 6 * f_Ps)

        return Flux*(Line + ratio*oPs)

        
