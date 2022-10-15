import numpy as np
import astropy.units as u

__all__ = ['power_spectrum']


@u.quantity_input(d=u.s)
def power_spectrum(flux, d=60 * u.s):
    """
    Compute the power spectrum of ``fluxes`` in units of [ppm^2 / microHz].

    Parameters
    ----------
    flux : ~numpy.ndarray or ~astropy.units.Quantity
        Fluxes with zero mean.
    d : ~astropy.units.Quantity
        Time between samples.

    Returns
    -------
    freq : ~astropy.units.Quantity
        Frequencies in microHz
    power : ~astropy.units.Quantity
        Power at each frequency in units of [ppm^2 / microHz]
    """
    if hasattr(flux, 'unit'):
        # if flux has units, use them!
        flux_unit = flux.unit
        flux_unscaled = flux.to(u.dimensionless_unscaled).value
    else:
        # otherwise, assume fluxes are not rescaled
        flux_unit = u.dimensionless_unscaled
        flux_unscaled = flux

    fft = np.fft.rfft(flux_unscaled, norm="backward")
    power = np.real(fft * np.conj(fft)) * d.to(1 / u.uHz) / len(flux_unscaled)
    freq = np.fft.rfftfreq(len(flux_unscaled), d.to(1 / u.uHz))

    return freq, power.to(flux_unit ** 2 / u.uHz)
