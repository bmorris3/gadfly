import warnings

import numpy as np

import astropy.units as u
from astropy.modeling.models import Voigt1D
from astropy.utils.exceptions import AstropyUserWarning


__all__ = [
    'p_mode_amplitudes', 'delta_nu', 'nu_max',
    'tau_eff', 'fwhm', 'tau_gran',
    'granulation_amplitude', 'c_K'
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
    effective temperature.

    Given in the abstract of Ballot et al. (2011) [1]_,
    and Eqn 8 of Huber et al. (2011) [2]_.

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
    return (
        luminosity ** _huber_s /
        (mass ** _huber_t * temperature ** (_huber_r - 1) * c_K(temperature))
    )


@u.quantity_input(mass=u.g, temperature=u.K, luminosity=u.L_sun)
def p_mode_amplitudes(mass, temperature, luminosity):
    """
    p-mode oscillation power amplitudes.

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
    Large frequency separation scaling.

    Huber et al. (2012) Eqn 3 [1]_.

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
    Frequency of maximum power.

    Huber et al. (2012) Eqn 4 [1]_.

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
    Characteristic granulation timescale.

    Relation from  Kallinger et al. (2014) [1]_,
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
    Scale the FWHM of p-mode oscillation peaks in power.

    From the fit in Figure 7 of Corsaro et al. (2015) [1]_.

    Actual parameters for the fit from Enrico Corsaro
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
    if temperature < 4900 * u.K:
        if not quiet:
            message = (
                "The p-mode FWHM scaling relations from Corsaro "
                "et al. (2015) are only valid for "
                "effective temperatures >4900 K, but you "
                f"gave temperature={temperature}. The algorithm will proceed "
                f"by fixing temperature=4900 K within the calculation "
                f"for the p-mode FWHM scaling (only)."
            )
            warnings.warn(message, AstropyUserWarning)
        temperature = 4900 * u.K

    return float(
        _fwhm(_solar_temperature) / _fwhm(temperature)
    )


def _tau_gran(mass, temperature, luminosity):
    return luminosity / (mass * temperature ** 3.5)


@u.quantity_input(nu_max=u.uHz)
def tau_gran(mass, temperature, luminosity):
    """
    Granulation timescale scaling.

    Kjeldsen & Bedding (2011) Eqn 9 [1]_.

    Parameters
    ----------
    nu_max : ~astropy.units.Quantity
        Peak p-mode frequency

    References
    ----------
    .. [1] `Kjeldsen & Bedding (2011)
       <https://ui.adsabs.harvard.edu/abs/2011A%26A...529L...8K/abstract>`_
    """
    return float(
        _tau_gran(mass, temperature, luminosity) /
        _tau_gran(_solar_mass, _solar_temperature, _solar_luminosity)
    )


@u.quantity_input(radius=u.m)
def n(mass, radius, temperature, luminosity):
    """
    Proportion of granules on the stellar surface.

    From Kjeldsen & Bedding (2011) Eqn 13 [1]_.

    References
    ----------
    .. [1] `Kjeldsen & Bedding (2011)
       <https://ui.adsabs.harvard.edu/abs/2011A%26A...529L...8K/abstract>`_
    """
    return (radius / H_P(mass, temperature, luminosity)) ** 2


@u.quantity_input(mass=u.g, temperature=u.K, luminosity=u.L_sun)
def H_P(mass, temperature, luminosity):
    """
    Pressure scale height of stellar photosphere.

    Kjeldsen & Bedding (2011) Eqn 8 [1]_.

    Parameters
    ----------
    nu_max : ~astropy.units.Quantity
        p-mode maximum frequency
    mass : ~astropy.units.Quantity
        Stellar mass
    temperature : ~astropy.units.Quantity
        Effective temperature
    luminosity : ~astropy.units.Quantity
        Stellar luminosity

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


def _granulation_power_factor(mass, temperature, luminosity):
    return luminosity ** 2 / (mass ** 3 * temperature ** 5.5)


@u.quantity_input(mass=u.g, temperature=u.K, luminosity=u.L_sun)
def granulation_amplitude(mass, temperature, luminosity):
    """
    Granulation amplitude scaling.

    Kjeldsen & Bedding (2011) Eqn 24 [1]_.

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
    .. [1] `Kjeldsen & Bedding (2011)
       <https://ui.adsabs.harvard.edu/abs/2011A%26A...529L...8K/abstract>`_
    """
    return float(
        _granulation_power_factor(mass, temperature, luminosity) /
        _granulation_power_factor(
            _solar_mass, _solar_temperature, _solar_luminosity
        )
    )


@u.quantity_input(freq=u.uHz)
def _v_osc_kiefer(freq):
    """
    Velocity spectral power from Kiefer et al. (2018) [1]_.

    Parameters
    ----------
    freq : ~astropy.units.Quantity
        Frequency (not angular)

    References
    ----------
    .. [1] See Eqns 1-5 and the parameter estimates in Table 1 of `Kiefer et al. (2018)
       <https://ui.adsabs.harvard.edu/abs/2018SoPh..293..151K/abstract>`_

    """
    nu_max = 3079.76 * u.uHz  # peak p-mode frequency
    sigma = 181.8 * u.uHz  # stddev of Gaussian
    gamma = 150.9 * u.uHz  # HWHM of Lorentzian
    Sigma = 611.8 * u.uHz  # FWHM of Voigt
    S = -0.100  # asymmetry parameter
    a = 3299 * 1e4 * u.m**2 / u.s**2 / u.Hz  # height factor
    b = -581 * u.m**2 / u.s**2 / u.Hz  # offset factor
    A = 1 / np.pi * (np.arctan(S * (freq - nu_max) / Sigma).to(u.rad).value + 0.5)
    voigt_profile = Voigt1D(x_0=nu_max, amplitude_L=a, fwhm_L=2*gamma, fwhm_G=2.355 * sigma)
    return (A * (b + voigt_profile(freq)) * u.uHz).to(u.m**2/u.s**2).value


@u.quantity_input(freq=u.uHz, wavelength=u.nm, temperature=u.K)
def p_mode_intensity_kjeldsen_bedding(freq, wavelength=650 * u.nm, temperature=5777 * u.K):
    """
    Power spectrum intensity scaling for p-modes.

    Computes the amplitudes of p-mode oscillations in the velocity
    power spectrum from Kiefer et al. (2018) [1]_, and converts
    the velocity power spectrum to an intensity estimate using
    Kjeldsen & Bedding (1995) [1]_

    Parameters
    ----------
    freq : ~astropy.units.Quantity
        Frequency (not angular)
    wavelength : ~astropy.units.Quantity
        Wavelength of observations
    temperature : ~astropy.units.Quantity
        Effective temperature

    References
    ----------
    .. [1] See Eqns 1-5 and the parameter estimates in Table 1 of `Kiefer et al. (2018)
       <https://ui.adsabs.harvard.edu/abs/2018SoPh..293..151K/abstract>`_
    .. [2] See Eqn 5 of `Kjeldsen & Bedding (1995)
       <https://ui.adsabs.harvard.edu/abs/1995A%26A...293...87K/abstract>`_
    """
    return 20.1 * (_v_osc_kiefer(freq) /
        (wavelength / (550 * u.nm)) /
        (temperature / (5777 * u.K)) ** 2
    ).decompose() * u.cds.ppm**2 / u.uHz
