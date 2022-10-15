import numpy as np
import astropy.units as u

doc = "A unit representing parts-per-million"  # ...is useful here
ppm = u.def_unit(
    'ppm', (100 * u.percent) / 1e6, doc=doc
)
u.add_enabled_units(ppm)

__all__ = ['power_spectrum']


@u.quantity_input(quantity=1/u.uHz)
def to_psd_units(quantity):
    return quantity.to(ppm**2 / u.uHz)


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

    return freq.to(u.uHz), power.to(flux_unit ** 2 / u.uHz)


def spectral_binning(y, all_x, all_y):
    """
    Spectral binning via trapezoidal approximation.
    """
    min_ind = np.argwhere(all_y == y[0])[0, 0]
    max_ind = np.argwhere(all_y == y[-1])[0, 0]
    return np.trapz(y, all_x[min_ind:max_ind + 1]) / (all_x[max_ind] - all_x[min_ind])


def spectral_binning_err(y, all_x, all_y, constant=5):
    """
    Approximate uncertainties for spectral bins estimated
    from a solar/stellar power spectrum.
    """
    min_ind = np.argwhere(all_y == y[0])[0, 0]
    max_ind = np.argwhere(all_y == y[-1])[0, 0]
    mean_x = all_x[min_ind:max_ind + 1].mean()

    # This term scales down the stddev (uncertainty) by the root of the
    # number of points in the bin == Gaussian uncertainty
    gaussian_term = np.std(y) / len(y) ** 0.5

    # This term scales the uncertainty with the spectral resolution of the bin
    non_gaussian_term = mean_x / (all_x[max_ind] - all_x[min_ind]) / constant

    return gaussian_term * non_gaussian_term


def bin_power_spectrum(freq, power, log_freq_bins=True, **kwargs):
    """
    Bin a power spectrum, with log-spaced frequency bins.

    Parameters
    ----------
    freq : ~astropy.units.Quantity
        Frequency
    power : ~astropy.units.Quantity
        Power spectrum
    log_freq_bins : True
        Other options not implemented.

    Returns
    -------
    freq_bins : ~astropy.units.Quantity
    power_bins : ~astropy.units.Quantity
    power_bins_err : ~astropy.units.Quantity
    """
    from scipy.stats import binned_statistic

    if not log_freq_bins:
        raise NotImplementedError("`bin_power_spectrum` has only been "
                                  "vetted with `log_freq_bins=True`.")

    log_freq = np.log10(freq[1:].value)

    # Set the number of log-spaced frequency bins
    n_bins = len(log_freq) // 10000

    # Bin the power spectrum:
    bs = binned_statistic(
        log_freq, power[1:].value,
        statistic=lambda y: spectral_binning(
            y, all_x=freq.value[1:], all_y=power[1:].value
        ),
        bins=n_bins
    )

    # Compute the error in the power spectrum bins
    bs_err = binned_statistic(
        log_freq, power[1:].value,
        statistic=lambda y: spectral_binning_err(
            y, all_x=freq.value[1:], all_y=power[1:].value, **kwargs
        ),
        bins=n_bins
    )

    freq_bins = 10 ** (0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])) * freq.unit
    power_bins = to_psd_units(bs.statistic * power.unit)
    power_bins_err = to_psd_units(bs_err.statistic * power.unit)

    return freq_bins, power_bins, power_bins_err


def linear_space_to_jax_parameterization(all_S0s, all_omegas):
    """
    Convert from (linear-space) S0 and w0 coordinates to the
    "differential" coordinates used within the jax minimizer.
    """
    omega_diffs = [all_omegas[0]]
    for i in range(1, len(all_omegas)):
        omega_diffs.append(np.log10(all_omegas[i] - all_omegas[i - 1]))

    S0_diffs = [all_S0s[-1]]
    for i in np.arange(1, len(all_S0s))[::-1]:
        S0_diffs.append(np.log10(all_S0s[i - 1] - all_S0s[i]))
    S0_diffs = S0_diffs[::-1]

    initp = np.array(list(map(
        lambda x: np.array(x, dtype=np.float64),
        [S0_diffs, omega_diffs]
    ))).T.flatten()
    return initp


def jax_parameterization_to_linear_space(p):
    """
    Convert the best-fit jax parameters into the
    (linear-space) S0 and w0 parameters that celerite2 uses.
    """
    delta_S0_0, w0_0 = p[0:2]
    delta_S0_1, delta_w0_1 = p[2:4]
    delta_S0_2, delta_w0_2 = p[4:6]
    delta_S0_3, delta_w0_3 = p[6:8]
    S0_4, delta_w0_4 = p[8:10]

    S0_3 = 10 ** delta_S0_3 + S0_4
    S0_2 = 10 ** delta_S0_2 + S0_3
    S0_1 = 10 ** delta_S0_1 + S0_2
    S0_0 = 10 ** delta_S0_0 + S0_1

    w0_1 = w0_0 + 10 ** delta_w0_1
    w0_2 = w0_1 + 10 ** delta_w0_2
    w0_3 = w0_2 + 10 ** delta_w0_3
    w0_4 = w0_3 + 10 ** delta_w0_4

    all_S0s = np.array([S0_0, S0_1, S0_2, S0_3, S0_4])
    all_omegas = np.array([w0_0, w0_1, w0_2, w0_3, w0_4])
    return all_S0s, all_omegas


def linear_space_to_dicts(S0s, omegas, fixed_Q):
    """
    Create a list of dictionaries for SHOTerm kwargs
    out of linear-space S0 and w0 coordinates
    """
    result = []
    for S0, w0 in zip(S0s, omegas):
        result.append(
            dict(
                S0=S0,
                w0=w0,
                Q=fixed_Q
            )
        )
    return result
