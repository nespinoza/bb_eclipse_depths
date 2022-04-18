import numpy as np
import matplotlib.pyplot as plt

from astropy import units as q

import utils

# First, define properties of the system. First, stellar properties:
jmag = 13.
Mstar = 0.518 # Msun
Rstar = 0.01310 # Rsun
Tstar = 4700. # Kelvin

# Properties of the planet/system:
Rplanet = 1. # Earth units
a = 0.005 # AU

# General properties of the calculation:
Neclipses = 25 # number of eclipses
min_w = 0.3 
max_w = 30.

# Generate plot. First, get equilibrium temperature of the planet:
Rstar = Rstar * q.Rsun
a = a * q.AU

Teq = Tstar * np.sqrt( Rstar.to(q.AU).value / (2. * a.value) )
# Now, calculate model transit depths. To this end, first get Rp/Rs:
Rplanet = Rplanet * q.Rearth
rprs = Rplanet.to(q.Rsun) / Rstar

wavelengths = np.linspace(min_w, max_w, 1000)

model_depths = utils.get_model_depths(wavelengths, rprs.value, Tstar, Teq)
plt.plot(wavelengths, model_depths, color = 'black')

# Get the sigma_phot:
central_wavelengths, sigma_phot, texp = utils.get_sigma_phot(jmag = 13.)

# Get Nin:
Nin = utils.get_Nin(texp, a.value, Rplanet.value, Mstar)

# Calculate sigma-depth for all photometric filters:
sigma_depth = utils.sigma_depth(sigma_phot, Nin, Neclipses)

# Plot 3-sigma values:
plt.plot(central_wavelengths, 3. * sigma_depth, 'o', color = 'cornflowerblue')

plt.xlabel('Wavelength (microns)')
plt.ylabel('Eclipse depth (ppm)')
plt.show()
