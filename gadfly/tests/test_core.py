from astropy.constants import L_sun, M_sun, R_sun
import numpy as np
import astropy.units as u

from ..core import generate_solar_fluxes, generate_stellar_fluxes


def test_sun():

    x_sun, y_sun, kernel_sun = generate_solar_fluxes(
        duration=10*u.min, cadence=60*u.s
    )

    f = np.logspace(np.log10(1e-2 * 1e-6), np.log10(1e4 * 1e-6), 100000)

    M = 1*M_sun
    L = 1*L_sun
    T_eff = 5777 * u.K
    R = 1*R_sun
    x_sun, y_sun, kernel_sun2 = generate_stellar_fluxes(
        10*u.min, M, T_eff, R, L, cadence=60*u.s
    )

    np.testing.assert_allclose(
        kernel_sun.get_psd(2*np.pi*f),
        kernel_sun2.get_psd(2*np.pi*f)
    )
