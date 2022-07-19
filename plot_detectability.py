import astropy.units as units
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize

import utils

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "cm"


def plot_contours(star, n_eclipses, global_colorbar=False):

    # Independent variables in colormaps.
    a_in_au = np.geomspace(0.001, 0.1, 500)
    r_p_in_rearth = np.linspace(1., 20., 500)
    a, r_p = np.meshgrid(a_in_au, r_p_in_rearth)

    # Stuff for computing sigma depth.
    wavelengths, sigma_phot, texp = utils.get_sigma_phot(jmag=star["Jmag"])
    n_in = utils.get_Nin(texp, a, r_p, star["Mass"])

    # Make plot grid.
    nrows = 3
    ncols = 3
    n = len(sigma_phot)
    fig, axs = plt.subplots(nrows=nrows,
                            ncols=ncols,
                            sharex=True,
                            sharey=True,
                            figsize=(13, 10))
    fig.subplots_adjust(hspace=0.1)

    # Manually scale all plots to given range.
    normalizer = Normalize(0, 150)

    # Factor out contour levels if global colorbar wanted.
    cs = None
    for j in range(nrows):
        for k in range(ncols):
            ax = axs[j][k]

            index = ncols * j + k
            if (index < n):
                # Sigma depth from instrumentation.
                wavelength = wavelengths[index]
                sigma_depth = utils.sigma_depth(sigma_phot[index], n_in,
                                                n_eclipses)

                # Theoretical depth.
                planet_temperature = utils.equilibrium_temperature(star, a)
                rprs = ((r_p * units.Rearth).to(units.Rsun) /
                        star["Radius"]).value
                model_depth = utils.get_model_depths(wavelength, rprs,
                                                     star["Temperature"],
                                                     planet_temperature)

                # 3-sigma percentage of detectability.
                detectability = (model_depth / (sigma_depth)).value

                if np.any(detectability > 3.):
                    l = ax.contour(a,
                                   r_p,
                                   detectability,
                                   colors='white',
                                   levels=[3.])
                    ax.clabel(l, fmt="%1.0f", fontsize=14)

                cs = ax.contourf(a,
                                 r_p,
                                 detectability,
                                 cmap='hot',
                                 norm=normalizer,
                                 levels=25)

                if k == 0:
                    ax.set_ylabel(r"$R_\mathrm{p}\,[R_\oplus]$", fontsize=20)

                ax.set_xscale('log')
                ax.set_yscale('log')
                plt.setp(ax.get_xticklabels(), fontsize=18)
                plt.setp(ax.get_yticklabels(), fontsize=18)

                ax.text(
                    0.13 * a_in_au.max(),
                    1.2 * r_p_in_rearth.min(),
                    r"$\lambda = {:1.2f}\,\mu\mathrm{{m}}$".format(wavelength),
                    fontsize=12,
                    bbox=dict(edgecolor='black',
                              facecolor='white',
                              boxstyle='round'))

                if not global_colorbar:
                    cbar = fig.colorbar(cs, ax=ax)
                    cbar.ax.tick_params(labelsize=14)

            else:
                ax.axis('off')

    for j in range(ncols):
        axs[nrows - 1][j].set_xlabel(r"$a$ [AU]", fontsize=20)

    if global_colorbar:
        # To-do: `cs` is the last contourf plotted. We want a global range!
        print("Warning: global colorbar showing scale of last contourf.")
        fig.colorbar(cs, ax=axs.ravel().tolist(), cmap='hot')

    plt.savefig("DetectabilityAtJmag{:1.0f}.pdf".format(star["Jmag"]),
                bbox_inches="tight")


if __name__ == "__main__":

    star = {}
    star["Jmag"] = 13.
    star["Mass"] = 0.518
    star["Radius"] = 0.01310
    star["Temperature"] = 4700.

    n_eclipses = 25

    plot_contours(star, n_eclipses)
