import numpy as np
import astropy.units as u
from astropy.units import cds  # noqa
from astropy.timeseries import LombScargle

from .interp import interpolate_missing_data

# aliased here for convenience
ppm = u.cds.ppm

__all__ = ['PowerSpectrum', 'plot_power_spectrum']


def to_psd_units(power, assumed_flux_unit=ppm, assumed_freq_unit=u.uHz):
    """
    Return ``power`` with units of ppm^2/uHz.
    If ``power`` has no units, assume it has units of
    ``[ppm^2 / uHz]``.
    """
    if not hasattr(power, 'unit'):
        power = power * assumed_flux_unit**2 / assumed_freq_unit
    return power.to(ppm**2 / u.uHz)


def to_freq_units(freq, assumed_freq_unit=u.uHz):
    """
    Return ``freq`` with units of uHz.
    If ``freq`` has no units, assume it has units of
    ``[uHz]``.
    """
    if not hasattr(freq, 'unit'):
        freq = freq * assumed_freq_unit
    return freq.to(assumed_freq_unit)


def plot_power_spectrum(
    ax=None,
    kernel=None,
    obs=None,
    freq=None,
    figsize=(8, 4),
    n_samples=1000,
    p_mode_inset=False,
    legend=True,
    scaling_low_freq='loglog',
    scaling_p_mode='semilogy',
    inset_xlim=[1800, 4500],
    inset_ylim=[0.005, 1.3],
    title=None,
    label_kernel=None,
    label_obs=None,
    label_inset='p-modes',
    kernel_kwargs=dict(),
    obs_kwargs=dict(),
    inset_kwargs=dict(),
    create_new_figure=False
):
    """
    Plot a power spectrum.

    Requires ``matplotlib``.

    Parameters
    ----------
    ax : :py:class:`~matplotlib.axes.Axes`
    kernel : None or subclass of :py:class:`~celerite2.terms.Term`
    obs : ~gadfly.psd.PowerSpectrum
    freq : ~astropy.units.Quantity
    figsize : list of floats
    n_samples : int
    p_mode_inset : bool
    legend : bool
    scaling_low_freq : str
    scaling_p_mode : str
    inset_xlim : list of floats
    inset_ylim : list of floats
    title : str
    label_inset : str
    label_obs : str
    label_kernel : str
    kernel_kwargs : dict
    obs_kwargs : dict
    inset_kwargs : dict
    create_new_figure : bool

    Returns
    -------
    fig : :py:class:`~matplotlib.figure.Figure`
    ax : :py:class:`~matplotlib.axes.Axes`
    """
    if obs is None and kernel is None:
        raise ValueError("Requires an observed power spectrum, a kernel, or both.")

    import matplotlib.pyplot as plt

    if freq is None:
        frequencies_all = np.logspace(-1, 3.5, int(n_samples) // 2) * u.uHz
        frequencies_p_mode = np.linspace(2000, 4500, int(n_samples) // 2) * u.uHz

        freq = np.sort(
            np.concatenate([frequencies_all, frequencies_p_mode])
        )

    if ax is None and create_new_figure:
        fig, ax = plt.subplots(figsize=figsize)
    elif ax is None:
        ax = plt.gca()

    fig = plt.gcf()

    if p_mode_inset:
        ax_inset = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    else:
        ax_inset = None

    # Set default values, then overwrite them if supplied by user:
    obs_kw = dict(marker='o', lw=0, mfc='none')
    for k, v in obs_kwargs.items():
        obs_kw[k] = v

    kernel_kw = dict(lw=0.5)
    for k, v in kernel_kwargs.items():
        kernel_kw[k] = v

    inset_kw = dict(marker='.', lw=0)
    for k, v in inset_kwargs.items():
        inset_kw[k] = v

    for i, (axis, plot_method, obs_plot_kwargs) in enumerate(zip(
        [ax, ax_inset],
        [scaling_low_freq, scaling_p_mode],
        [obs_kw, inset_kw]
    )):
        if axis is not None:
            if kernel is not None:
                if label_kernel is None:
                    if kernel.name is None:
                        label_kernel = 'Model'
                    else:
                        label_kernel = kernel.name
                # call the plot method (i.e.: plt.semilogy, plt.loglog)
                getattr(axis, plot_method)(
                    to_freq_units(freq),
                    to_psd_units(kernel.get_psd(2 * np.pi * freq.to(u.uHz).value)),
                    label=label_kernel, **kernel_kw
                )
            if obs is not None:
                if obs.name is not None and label_obs is None:
                    label_obs = obs.name
                elif label_obs is None:
                    label_obs = 'Observations'

                # call the plot method (i.e.: plt.semilogy, plt.loglog)
                getattr(axis, plot_method)(
                    to_freq_units(obs.frequency),
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
            label_inset, (0.97 * inset_xlim[1], 0.5 * inset_ylim[1]),
            ha='right'
        )
        ax.indicate_inset_zoom(ax_inset, edgecolor="silver")

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

    if max_ind > min_ind:
        gaussian_term = np.nanstd(y) / len(y) ** 0.5
        # This term scales the uncertainty with the spectral resolution of the bin
        non_gaussian_term = mean_x / (all_x[max_ind] - all_x[min_ind]) / constant

        return gaussian_term * non_gaussian_term

    return np.nan


def bin_power_spectrum(power_spectrum, bins=None, log=True, **kwargs):
    """
    Bin a power spectrum, with log-spaced frequency bins.

    Parameters
    ----------
    power_spectrum : ~gadfly.PowerSpectrum
    log : bool
        If true, compute bin edges based on the log base 10 of
        the frequency.
    bins : int or ~numpy.ndarray
        Number of bins, or the bin edges

    Returns
    -------
    new_ps : ~gadfly.PowerSpectrum
    """
    from scipy.stats import binned_statistic
    freq = power_spectrum.frequency
    power = power_spectrum.power

    if log:
        freq_axis = np.log10(freq.to(u.uHz).value)
    else:
        freq_axis = freq.to(u.uHz).value

    # Set the number of log-spaced frequency bins
    if bins is None:
        bins = len(freq_axis) // 10000

    # Bin the power spectrum:
    bs = binned_statistic(
        freq_axis, power.value,
        statistic=lambda y: spectral_binning(
            y, all_x=freq_axis, all_y=power.value
        ),
        bins=bins
    )

    # Compute the error in the power spectrum bins
    bs_err = binned_statistic(
        freq_axis, power.value,
        statistic=lambda y: spectral_binning_err(
            y, all_x=freq_axis, all_y=power.value, **kwargs
        ),
        bins=bins
    )

    if log:
        freq_bins = 10 ** (
            0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])
        ) * freq.unit
    else:
        freq_bins = (
            0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])
        ) * freq.unit

    power_bins = to_psd_units(bs.statistic * power.unit)
    power_bins_err = to_psd_units(bs_err.statistic * power.unit)

    name = (
        power_spectrum.name if power_spectrum.name is not None
        else 'Power spectrum'
    ) + ' (binned)'

    return PowerSpectrum(
        freq_bins, power_bins, power_bins_err,
        name=name
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
    def __init__(self, frequency, power, error=None, name=None, norm=None, detrended_lc=None):
        self.frequency = frequency
        self.power = power
        self.error = error
        self.name = name
        self.norm = norm
        self.detrended_lc = detrended_lc

    @property
    def omega(self):
        """
        Angular frequency.

        Returns a numpy array of :math:`\\omega = 2 \\pi f`
        with f in units of [uHz].

        Returns
        -------
        omegas : ~numpy.ndarray
        """
        return 2 * np.pi * self.frequency.to(u.uHz).value

    @property
    def light_curve_rms(self):
        """
        Convert from power spectral density to the approximate
        root mean square of the original light curve.

        Returns
        -------
        rms : ~astropy.units.Quantity
        """
        return (self.power * self.norm).to(ppm**2) ** 0.5

    def bin(self, bins=None, **kwargs):
        """
        Bin the power spectrum.

        Requires scipy.

        Parameters
        ----------
        bins : int or ~numpy.ndarray
            Number of bins or an array of bin edges.

        Returns
        -------
        new_ps : ~gadfly.PowerSpectrum
            A new, binned power spectrum
        """
        return bin_power_spectrum(self, bins, **kwargs)

    @classmethod
    def from_light_curve(
        cls, light_curve, method='fft', include_zero_freq=False, name=None,
        detrend=True, detrend_poly_order=3, save_detrended_lc=True
    ):
        """
        Compute the power spectrum from a light curve.

        Parameters
        ----------
        light_curve : ~lightkurve.lightcurve.LightCurve, ~lightkurve.lightkurve.LightCurveCollection # noqa
            Light curve(s)
        method : str, options are "fft" or "lomb-scargle"
            Method used to compute the power spectrum.
        include_zero_freq : bool
            Include ``frequency=0`` in the first entry of the results.
        name : str
            Name for the power spectrum
        detrend : bool
        detrend_poly_order : int
        save_detrended_lc : bool
        """
        from lightkurve import LightCurve, LightCurveCollection

        available_methods = ['fft', 'lomb-scargle']
        if not method.lower() in available_methods:
            raise ValueError(f'PowerSpectrum.from_lightcurve was given '
                             f'method="{method}", but it must be '
                             f'one of: {available_methods}.')

        # Interpolation is required if the PSD computation method is FFT
        method_is_fft = method.lower() == 'fft'

        if not isinstance(light_curve, LightCurveCollection):
            light_curve = LightCurveCollection([light_curve])

        if detrend:
            # Interpolate over missing data points in each quarter, normalize by a
            # nth order polynomial to remove systematic trends
            lcs_interped = []
            for lc in light_curve:
                lc_flux_unit = lc.flux.unit
                lc = lc.remove_nans().remove_outliers()
                if method_is_fft:
                    t, f = interpolate_missing_data(lc.time.jd, lc.flux.value)
                else:
                    t, f = lc.time.jd, lc.flux.value
                e = np.median(lc.flux_err) * np.ones_like(f)

                # if the light curve is not given in units combatible with ppm,
                # normalize it with a polynomial, center it at zero
                if not lc_flux_unit.is_equivalent(ppm):
                    fit = np.polyval(
                        np.polyfit(t - t.mean(), f, detrend_poly_order),
                        t - t.mean()
                    )
                    normed_flux = f / fit
                    flux_ppm = 1e6 * np.array(normed_flux / np.median(normed_flux) - 1) * ppm
                else:
                    flux_ppm = f.copy()

                lc_int = LightCurve(
                    time=t,
                    # convert detrended flux to ppm with zero-mean:
                    flux=flux_ppm,
                    flux_err=e.value
                )
                lcs_interped.append(lc_int)

            if len(lcs_interped) > 1:
                slc = LightCurveCollection(lcs_interped).stitch(lambda x: x)
            else:
                slc = lcs_interped[0]

            if method_is_fft:
                # After stitching together all quarters, interpolate again to fill gaps
                # between the quarters:
                t, f = interpolate_missing_data(slc.time.jd, slc.flux.value)
            else:
                t, f = slc.time.jd, slc.flux.value

            d = (t[1] - t[0]) * u.d
            flux = f.copy() * ppm
            time = t.copy() * u.day
            name = light_curve[0].meta.get('name', name)
        else:
            if isinstance(light_curve, LightCurveCollection):
                light_curve = light_curve.stitch(lambda x: x)

            d = (light_curve.time[1] - light_curve.time[0]).to(u.d)
            time = light_curve.time.jd * u.day
            flux = light_curve.flux
            name = light_curve.meta.get('name', name)

        if not hasattr(flux, 'unit'):
            # if flux has no units, assume ppm:
            flux = flux * ppm

        if method_is_fft:
            freq, power, norm = cls._fft(flux, d)
        else:
            freq, power, norm = cls._lomb_scargle(time, flux, d)

        # skip `frequency==0` if necessary:
        if not include_zero_freq:
            freq = freq[1:]
            power = power[1:]

        detrended_lc = LightCurve(time=time, flux=flux) if (detrend and save_detrended_lc) else None

        return cls(freq, power, name=name, norm=norm, detrended_lc=detrended_lc)

    @staticmethod
    def _fft(flux, d):
        """
        Compute the power spectrum using the FFT. Inputs and outputs
        are ~astropy.units.Quantity objects.
        """
        # Strip units from flux:
        flux_ppm = flux.to(ppm).value

        # Measure the observed power spectrum via FFT:
        freq = np.fft.rfftfreq(len(flux_ppm), d).to(u.uHz)
        fft = np.fft.rfft(flux_ppm)

        # The FFT must be scaled by this factor, in addition
        # to the normalization by the length of the flux vector:
        norm = d / (2 * np.pi) ** 0.5 / len(flux_ppm)

        # The power spectrum in the usual asteroseismic units:
        power = to_psd_units(
            np.real(fft * np.conj(fft)) * ppm ** 2 * norm
        )
        return freq, power, norm

    @staticmethod
    def _lomb_scargle(time, flux, d):
        """
        Compute the power spectrum using the Lomb-Scargle method.
        Inputs and outputs are ~astropy.units.Quantity objects.
        """
        # use the rfftfreq method to get frequencies for consistency with fft method:
        freq = np.fft.rfftfreq(len(flux), d).to(u.uHz)

        lomb_scargle = LombScargle(time, flux, normalization='psd')
        norm = d / (2 * np.pi) ** 0.5
        power = to_psd_units(norm * lomb_scargle.power(freq))
        return freq, power, norm

    def plot(self, **kwargs):
        """
        See docstring for :py:func:`~gadfly.plot_power_spectrum` for arguments
        """
        return plot_power_spectrum(
            obs=self, **kwargs
        )

    def cutout(self, frequency_min=None, frequency_max=None):
        """
        Cut out a section of the power spectrum.

        Parameters
        ----------
        frequency_min : ~astropy.units.Quantity
            Cut out any measurements below this frequency
        frequency_max : ~astropy.units.Quantity
            Cut out any measurements above this frequency

        Returns
        -------
        new_ps : ~gadfly.PowerSpectrum
            A new, cropped power spectrum
        """
        if frequency_min is None:
            frequency_min = 0 * u.Hz
        if frequency_max is None:
            frequency_max = np.inf * u.Hz

        bounds = (
            (self.frequency <= frequency_max) &
            (self.frequency >= frequency_min)
        )

        name = (
            self.name if self.name is not None
            else 'Power spectrum'
        ) + ' (cutout)'

        args = []
        if self.error is not None:
            args.append(self.error[bounds])

        return PowerSpectrum(
            self.frequency[bounds], self.power[bounds], *args, name=name, norm=self.norm
        )
