import numpy as np
import astropy.units as u

doc = "A unit representing parts-per-million"  # ...is useful here
ppm = u.def_unit(
    'ppm', (100 * u.percent) / 1e6, doc=doc
)
u.add_enabled_units(ppm)

__all__ = ['PowerSpectrum', 'plot_power_spectrum']


@u.quantity_input(quantity=1/u.uHz)
def to_psd_units(quantity):
    return quantity.to(ppm**2 / u.uHz)


def plot_power_spectrum(
    p_mode_inset=True, kernel=None, n_samples=1000,
    figsize=(8, 4), scaling_low_freq='loglog',
    scaling_p_mode='semilogy', inset_xlim=[1800, 4500],
    inset_ylim=[0.03, 1.3], title=None,
    label_kernel=None, label_obs=None, label_inset='p-modes',
    obs=None, kernel_kwargs=dict(color='r'), legend=True,
    obs_kwargs=dict(color='k', marker='o', lw=0),
    inset_kwargs=dict(color='k', marker='.', lw=0), ax=None
):
    """
    Plot a power spectrum.

    Parameters
    ----------
    p_mode_inset : bool
    kernel : None or subclass of ~celerite2.terms.Term
    n_samples : int
    figsize : list of floats
    scaling_low_freq : str
    scaling_p_mode : str
    inset_xlim : list of floats
    inset_ylim : list of floats
    title : str
    label_inset : str
    label_obs : str
    label_kernel : str
    obs : ~gadfly.psd.PowerSpectrum
    kernel_kwargs : dict
    legend : bool
    obs_kwargs : dict
    ax : ~matplotlib.axes.Axis

    Returns
    -------
    fig, ax
    """
    if obs is None and kernel is None:
        raise ValueError("Requires an observed power spectrum, a kernel, or both.")

    import matplotlib.pyplot as plt
    frequencies_all = np.logspace(-1, 3.5, int(n_samples) // 2) * u.uHz
    frequencies_p_mode = np.linspace(2000, 4500, int(n_samples) // 2) * u.uHz

    freq = np.sort(
        np.concatenate([frequencies_all, frequencies_p_mode])
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()

    if p_mode_inset:
        ax_inset = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    else:
        ax_inset = None

    for i, (axis, plot_method, obs_plot_kwargs) in enumerate(zip(
        [ax, ax_inset],
        [scaling_low_freq, scaling_p_mode],
        [obs_kwargs, inset_kwargs]
    )):
        if axis is not None:
            if kernel is not None:

                if label_kernel is None:
                    label_kernel = 'Model'

                getattr(axis, plot_method)(
                    freq, kernel.get_psd(2 * np.pi * freq.to(u.uHz).value),
                    label=label_kernel, **kernel_kwargs
                )
            if obs is not None:

                if label_obs is None:
                    label_obs = 'Observations'

                getattr(axis, plot_method)(
                    obs.frequency,
                    to_psd_units(obs.power),
                    label=label_obs, **obs_plot_kwargs
                )
            axis.set(
                xlabel='Frequency [$\\mu$Hz]',
                ylabel='Power [ppm$^2$ / $\\mu$Hz]',
            )
    if p_mode_inset:
        ax_inset.set_xlim(inset_xlim)
        ax_inset.set_ylim(inset_ylim)
        ax_inset.annotate(
            label_inset, (0.97 * inset_xlim[1], 0.7 * inset_ylim[1]),
            ha='right'
        )
        ax.indicate_inset_zoom(ax_inset, edgecolor="silver")

    if title is None:
        if obs is not None:
            title = obs.name
        elif kernel is not None:
            title = kernel.name
        else:
            title = 'Power spectrum'

    if legend:
        ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def spectral_binning(y, all_x, all_y):
    """
    Spectral binning via trapezoidal approximation.
    """
    min_ind = np.argwhere(all_y == y[0])[0, 0]
    max_ind = np.argwhere(all_y == y[-1])[0, 0]
    if max_ind > min_ind:
        return np.trapz(y, all_x[min_ind:max_ind + 1]) / (all_x[max_ind] - all_x[min_ind])
    return y[0]


def spectral_binning_err(y, all_x, all_y, constant=5):
    """
    Approximate uncertainties for spectral bins estimated
    from a solar/stellar power spectrum.
    """
    min_ind = np.argwhere(all_y == y[0])[0, 0]
    max_ind = np.argwhere(all_y == y[-1])[0, 0]
    mean_x = np.nanmean(all_x[min_ind:max_ind + 1])

    # This term scales down the stddev (uncertainty) by the root of the
    # number of points in the bin == Gaussian uncertainty
    gaussian_term = np.std(y) / len(y) ** 0.5

    if max_ind > min_ind:
        # This term scales the uncertainty with the spectral resolution of the bin
        non_gaussian_term = mean_x / (all_x[max_ind] - all_x[min_ind]) / constant
    else:
        non_gaussian_term = mean_x / constant

    return gaussian_term * non_gaussian_term


def bin_power_spectrum(power_spectrum, bins=None, **kwargs):
    """
    Bin a power spectrum, with log-spaced frequency bins.

    Parameters
    ----------
    freq : ~astropy.units.Quantity
        Frequency
    power : ~astropy.units.Quantity
        Power spectrum
    bins : int or ~numpy.ndarray
        Number of bins, or the bin edges

    Returns
    -------
    freq_bins : ~astropy.units.Quantity
    power_bins : ~astropy.units.Quantity
    power_bins_err : ~astropy.units.Quantity
    """
    from scipy.stats import binned_statistic
    freq = power_spectrum.frequency
    power = power_spectrum.power

    log_freq = np.log10(freq.value)

    # Set the number of log-spaced frequency bins
    if bins is None:
        bins = len(log_freq) // 10000

    # Bin the power spectrum:
    bs = binned_statistic(
        log_freq, power.value,
        statistic=lambda y: spectral_binning(
            y, all_x=freq.value, all_y=power.value
        ),
        bins=bins
    )

    # Compute the error in the power spectrum bins
    bs_err = binned_statistic(
        log_freq, power.value,
        statistic=lambda y: spectral_binning_err(
            y, all_x=freq.value, all_y=power.value, **kwargs
        ),
        bins=bins
    )

    freq_bins = 10 ** (
        0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])
    ) * freq.unit
    power_bins = to_psd_units(bs.statistic * power.unit)
    power_bins_err = to_psd_units(bs_err.statistic * power.unit)

    return PowerSpectrum(
        freq_bins, power_bins, power_bins_err,
        name=power_spectrum.name + " (binned)"
    )


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


class PowerSpectrum:
    """
    An observed power spectrum.
    """

    @u.quantity_input(frequency=u.uHz, power=ppm**2/u.uHz)
    def __init__(self, frequency, power, error=None, name=None):
        self.frequency = frequency
        self.power = power
        self.error = error
        self.name = name

    def bin(self, bins=None, **kwargs):
        """
        Bin the power spectrum, return a new PowerSpectrum.

        Parameters
        ----------
        bins : int or ~numpy.ndarray
            Number of bins or an array of bin edges.
        """
        return bin_power_spectrum(self, bins, **kwargs)

    @classmethod
    def from_light_curve(cls, light_curve, include_zero_freq=False, name=None):
        """
        Compute the power spectrum from a light curve.

        Parameters
        ----------
        light_curve : ~lightkurve.lightcurve.LightCurve
            Light curve
        include_zero_freq : bool
            Include ``frequency=0`` in the first entry of the results.
        name : str
            Name for the power spectrum
        """
        d = (light_curve.time[1] - light_curve.time[0]).to(u.d)
        flux = light_curve.flux

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

        if not include_zero_freq:
            freq = freq[1:]
            power = power[1:]

        name = light_curve.meta.get('name', name)

        return cls(freq.to(u.uHz), power.to(flux_unit ** 2 / u.uHz), name=name)

    def plot(self, **kwargs):
        """
        See docstring for :py:func:`~gadfly.plot_power_spectrum` for arguments
        """
        return plot_power_spectrum(
            obs=self, **kwargs
        )
