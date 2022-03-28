import numpy as np

from astropy.modeling.models import BlackBody
from astropy.constants import G
from astropy import units as q

def get_model_depths(wavelength, rprs, Tstar, Tplanet):
    """
    Function that given a wavelength (or set of wavelengths), the temperature
    of the star, the temperature of the planet and the planet-to-star radius 
    ratio, returns the model eclipse depth assuming only thermal emission 
    from the planet.

    Input
    -----

    wavelength : float or np.array
        Float or array of floats representing the wavelengths (in microns)

    rprs : float
        Planet-to-star radius ratio

    Tstar : float
        Temperature of the star in Kelvins.

    Tplanet : float 
        Temperature of the planet in Kelvins.

    Returns
    -------

    depth : float or np.array
        Eclipse depths in parts-per-million at the requested wavelength(s)
    
    """
    
    # Get blackbody functions for each component
    bb_star = BlackBody(Tstar * q.K)
    bb_planet = BlackBody(Tplanet * q.K) 
    
    # Get area ratio:
    ApAs = (rprs)**2

    # Get flux ratio:
    wavelength = wavelength * q.um
    FpFs = bb_planet(wavelength) / bb_star(wavelength) 

    if rprs >= 1.:

        FinFout = (1. + (FpFs) * (ApAs - 1.)) / (1. + FpFs * ApAs)

    else:

        FinFout = 1. / (1. + (FpFs * ApAs))

    return ( 1. - FinFout ) * 1e6

def get_sigma_phot(jmag = 13.):
    """
    Function that gets you back the sigma_phot for MIRI time-series photometry of a 
    white-dwarf. A star with Teff = 4700 was used.

    Input
    -----

    jmag : float
        J-magnitude of the star
    
    Returns
    -------

    wavelength: np.array
        Central wavelengths of the photometric filters

    sigma_phot : numpy array
        Photometric error in parts-per-million for each wavelength.

    texp : flat
        Integration time (in seconds).
    """

    # Central wavelength of the photometric filters used:
    central_wavelengths = np.array([5.56, 7.51, 9.86, 11.29, 
                                    12.68, 14.80, 17.82, 20.47, 
                                    24.96])

    texp = 17.02 # seconds:

    if jmag == 13.:

        # Define SNRs at these throughputs (obtained through the JWST ETC), 
        # i.e., https://jwst.etc.stsci.edu/
        sns = np.array([325.66, 340.78, 216.92, 96.05,
                        126.85, 74.42, 26.72, 11.67, 
                        1.83])

    elif jmag == 14.:

        # Define SNRs at these throughputs (obtained through the JWST ETC), 
        # i.e., https://jwst.etc.stsci.edu/
        sns = np.array([209.36, 208.80, 118.72, 48.71,
                        60.39, 33.09, 11.44, 4.94, 
                        0.77])

    elif jmag == 15.:

        # Define SNRs at these throughputs (obtained through the JWST ETC), 
        # i.e., https://jwst.etc.stsci.edu/
        sns = np.array([129.15, 116.30, 56.59, 21.65,
                        25.67, 13.52, 4.59, 1.97,  
                        0.31])

    else:

        raise Exception('Only magnitudes 13, 14 and 15 are allowed.')

    return central_wavelengths, (1. / sns) * 1e6, texp

def get_Nin(texp, a, Rp, Ms):
    """
    Function that gets you the number of in-eclipse datapoints given a set of 
    inputs. This assumes the mass of the planet is negligible to the mass of the star.

    Inputs
    ------

    texp : float
        Exposure time for each photometric datapoint.
    a : float
        Semi-major axis in AU.
    Rp : float
        Planetary radius in units of REarth
    Ms : flat
        Stellar mass in units of MSun
    """

    a = a * q.AU
    Rp = Rp * q.Rearth
    Ms = Ms * q.Msun
 
    # Get period from Kepler's Third law:
    P = np.sqrt( (4. * np.pi**2 ) * (a**3) / (G * Ms) )

    # Get ratios of planet radii and semi-major axis:
    rpa = Rp / a.to(q.Rearth)

    # Return Nin:
    return ( P.to(q.s).value / np.pi ) * rpa / texp
