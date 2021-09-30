import numpy as np

__all__ = ['power_spectrum']


def power_spectrum(fluxes, d=60):
    """
    Compute the power spectrum of ``fluxes`` in units of [ppm^2 / microHz].

    Parameters
    ----------
    fluxes : `~numpy.ndarray`
        Fluxes with zero mean.
    d : float
        Time between samples [s].

    Returns
    -------
    freq : `~numpy.ndarray`
        Frequencies
    power : `~numpy.ndarray`
        Power at each frequency in units of [ppm^2 / microHz]
    """
    fft = np.fft.rfft(1e6 * fluxes)
    power = (fft * np.conj(fft)).real * 1e-6 / len(fluxes)
    freq = np.fft.rfftfreq(len(fluxes), d)
    return freq, power
