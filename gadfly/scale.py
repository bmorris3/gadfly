import warnings

import numpy as np

import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning


__all__ = [
    'p_mode_amplitudes', 'delta_nu', 'nu_max',
    'tau_eff', 'fwhm', 'tau_gran',
    'granulation_amplitude'
]

# Solar parameters
_solar_temperature = 5777 * u.K
_solar_mass = 1 * u.M_sun
_solar_radius = 1 * u.R_sun
_solar_luminosity = 1 * u.L_sun

# Huber et al. (2011)
# https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract
_solar_nu_max = 3090 * u.uHz

# Bahng & Schwarzschild (1961)
# https://ui.adsabs.harvard.edu/abs/1961ApJ...134..312B/abstract
_solar_granulation_timescale = 8.6 * u.min

# https://www2.mps.mpg.de/solar-system-school/germerode/Solar-Photosphere-Chromosphere.pdf
_solar_pressure_scale_height = 125 * u.km

# scaling relation parameters from Huber et al. (2011)
# https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract
_huber_r = 2
_huber_s = 0.886
_huber_t = 1.89


@u.quantity_input(temperature=u.K)
def c_K(temperature):
    """
    Bolometric correction factor as a function of
    effective temperature, given in the abstract of
    Ballot et al. (2011) [1]_, and Eqn 8 of Huber et al.
    (2011) [2]_.

    Parameters
    ----------
    temperature : ~astropy.units.Quantity
        Effective temperature

    References
    ----------
    .. [1] `Ballot et al. (2011)
       <https://ui.adsabs.harvard.edu/abs/2011A%26A...531A.124B/abstract>`_

    .. [2] `Huber et al. (2011)
       <https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract>`_
    """
    return float(
        (temperature / (5934 * u.K)) ** 0.8
    )


def _p_mode_amplitudes(mass, temperature, luminosity):
    return float(
        float(luminosity / u.L_sun) ** _huber_s /
        (float(mass / u.M_sun) ** _huber_t *
         temperature.to(u.K).value ** (_huber_r - 1) *
         c_K(temperature))
    )


@u.quantity_input(mass=u.g, temperature=u.K, luminosity=u.L_sun)
def p_mode_amplitudes(mass, temperature, luminosity):
    """
    Huber et al. (2011), Eqn 9 [1]_.

    Parameters
    ----------
    mass : ~astropy.units.Quantity
        Stellar mass
    temperature : ~astropy.units.Quantity
        Effective temperature
    luminosity : ~astropy.units.Quantity
        Stellar luminosity

    References
    ----------
    .. [1] `Huber et al. (2011)
       <https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract>`_
    """
    return float(
        _p_mode_amplitudes(mass, temperature, luminosity) /
        _p_mode_amplitudes(_solar_mass, _solar_temperature, _solar_luminosity)
    )


@u.quantity_input(mass=u.g, radius=u.m)
def delta_nu(mass, radius):
    """
    Large frequency separation scaling. Huber et al. (2012) Eqn 3 [1]_.

    Parameters
    ----------
    mass : ~astropy.units.Quantity
        Stellar mass
    radius : ~astropy.units.Quantity
        Stellar radius

    References
    ----------
    .. [1] `Huber et al. (2012)
       <https://ui.adsabs.harvard.edu/abs/2012ApJ...760...32H/abstract>`_
    """
    return float(
        (mass / _solar_mass) ** 0.5 *
        (radius / _solar_radius) ** (-3 / 2)
    )


@u.quantity_input(mass=u.g, temperature=u.K, radius=u.m)
def nu_max(mass, temperature, radius):
    """
    Frequency of maximum power. Huber et al. (2012) Eqn 4 [1]_.

    Parameters
    ----------
    mass : ~astropy.units.Quantity
        Stellar mass
    temperature : ~astropy.units.Quantity
        Effective temperature
    radius : ~astropy.units.Quantity
        Stellar radius

    References
    ----------
    .. [1] `Huber et al. (2012)
       <https://ui.adsabs.harvard.edu/abs/2012ApJ...760...32H/abstract>`_
    """
    return float(
        (mass / _solar_mass) *
        (radius / _solar_radius) ** -2 *
        (temperature / _solar_temperature) ** -0.5
    )


@u.quantity_input(nu=u.Hz)
def tau_eff(nu_max):
    """
    Characteristic granulation timescale as in Kallinger et al. (2014) [1]_,
    described in the last sentence in the paragraph below Eqn 4.

    Parameters
    ----------
    nu_max : ~astropy.units.Quantity
        p-mode maximum frequency

    References
    ----------
    .. [1] `Kallinger et al. (2014)
       <https://ui.adsabs.harvard.edu/abs/2014A%26A...570A..41K/abstract>`_
    """
    return float(
        (nu_max / _solar_nu_max) ** -0.89
    )


def _fwhm(temperature):
    temperature_kelvin = temperature.to(u.K).value
    return np.exp(
        1463.49 -
        1.03503 * temperature_kelvin +
        0.000271565 * temperature_kelvin ** 2 -
        3.14139e-08 * temperature_kelvin ** 3 +
        1.35524e-12 * temperature_kelvin ** 4
    )


@u.quantity_input(temperature=u.K)
def fwhm(temperature, quiet=False):
    """
    Scale the natural log of the p-mode peaks' FWHM according to
    the fit in Figure 7 of Corsaro et al. (2015) [1]_.

    Actual parameterization from Enrico Corsaro
    (private communication).

    Parameters
    ----------
    temperature : ~astropy.units.Quantity
        Stellar temperature
    quiet : bool
        Set to True to raise a warning if the effective temperature
        is outside of the range where Corsaro's fit is valid.

    References
    ----------
    .. [1] `Corsaro et al. (2015)
       <https://ui.adsabs.harvard.edu/abs/2015A%26A...579A..83C/abstract>`_
    """
    if not quiet and temperature < 4900 * u.K:
        message = (
            "The p-mode FWHM scaling relations from Corsaro "
            "et al. (2015) are only valid for "
            "effective temperatures >4900 K, but you "
            f"gave temperature={temperature}. The algorithm will proceed "
            f"by fixing temperature=4900 K within the calculation "
            f"for the p-mode FWHM scaling (only)."
        )
        warnings.warn(message, AstropyUserWarning)

    return float(
        _fwhm(temperature) /
        _fwhm(_solar_temperature)
    )


@u.quantity_input(nu_max=u.uHz)
def tau_gran(nu_max):
    """
    Granulation timescale from Kjeldsen & Bedding (2011)
    Eqn 10 [1]_.

    Parameters
    ----------
    nu_max : ~astropy.units.Quantity
        Peak p-mode frequency

    References
    ----------
    .. [1] `Kjeldsen & Bedding (2011)
       <https://ui.adsabs.harvard.edu/abs/2011A%26A...529L...8K/abstract>`_
    """
    return _solar_nu_max / nu_max


@u.quantity_input(radius=u.m)
def n(mass, radius, temperature, luminosity):
    """
    Number of granules on the stellar surface
    from Kjeldsen & Bedding (2011) Eqn 13 [1]_.

    References
    ----------
    .. [1] `Kjeldsen & Bedding (2011)
       <https://ui.adsabs.harvard.edu/abs/2011A%26A...529L...8K/abstract>`_
    """
    return (radius / H_P(mass, temperature, luminosity)) ** 2


@u.quantity_input(mass=u.g, temperature=u.K, luminosity=u.L_sun)
def H_P(mass, temperature, luminosity):
    """
    Pressure scale height in Kjeldsen & Bedding (2011)
    Eqn 8 [1]_.

    Parameters
    ----------
    nu_max : ~astropy.units.Quantity
        p-mode maximum frequency

    References
    ----------
    .. [1] `Kjeldsen & Bedding (2011)
       <https://ui.adsabs.harvard.edu/abs/2011A%26A...529L...8K/abstract>`_
    """
    stellar_H_factor = luminosity / (mass * temperature ** 3)
    solar_H_factor = _solar_luminosity / (_solar_mass * _solar_temperature ** 3)
    return (
        _solar_pressure_scale_height * stellar_H_factor / solar_H_factor
    )


def _granulation_power_factor(nu, nu_max, mass, radius, temperature, luminosity):
    sigma_int = n(mass, radius, temperature, luminosity) ** -0.5
    tau_gran_quantity = _solar_granulation_timescale * tau_gran(nu_max)
    return (
        sigma_int ** 2 * tau_gran_quantity /
        (1 + (2 * np.pi * nu * tau_gran_quantity) ** 2)
    )


@u.quantity_input(nu=u.Hz, nu_max=u.Hz, mass=u.g, radius=u.m,
                  temperature=u.K, luminosity=u.L_sun)
def granulation_amplitude(nu, nu_max, mass, radius, temperature, luminosity):
    """
    Granulation amplitude scaling as in Kjeldsen & Bedding (2011)
    Eqn 21 [1]_.

    Parameters
    ----------
    nu : ~astropy.units.Quantity
        Frequency

    References
    ----------
    .. [1] `Kjeldsen & Bedding (2011)
       <https://ui.adsabs.harvard.edu/abs/2011A%26A...529L...8K/abstract>`_
    """
    return float(
        _granulation_power_factor(nu, nu_max, mass, radius, temperature, luminosity) /
        _granulation_power_factor(nu, _solar_nu_max, _solar_mass,
                                  _solar_radius, _solar_temperature, _solar_luminosity)
    )
