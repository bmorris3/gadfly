import os
import json
import warnings

import numpy as np

from celerite2 import terms as celerite2_terms

import astropy.units as u
from astropy.units import cds  # noqa
from astropy.utils.exceptions import AstropyUserWarning

import tynt

from . import scale
from .psd import plot_power_spectrum
from .sun import _p_mode_fit_to_sho_hyperparams

__all__ = [
    'Hyperparameters',
    'StellarOscillatorKernel',
    'SolarOscillatorKernel',
    'ShotNoiseKernel',
    'Filter'
]

dirname = os.path.dirname(os.path.abspath(__file__))
default_hyperparameter_path = os.path.join(
    dirname, 'data', 'hyperparameters.json'
)


def _sho_psd(omega, S0, w0, Q):
    """
    Stochastically driven, damped harmonic oscillator.
    """
    # This is the celerite2 SHO PSD:
    return (
        np.sqrt(2/np.pi) * S0 * w0**4 /
        ((omega**2 - w0**2)**2 + (omega**2 * w0**2 / Q**2))
    )


class Hyperparameters(list):
    """
    Gaussian process hyperparameters for approximating
    the total stellar irradiance power spectrum.
    """
    def __init__(self, hyperparameters, name=None, magnitude=None):
        """
        Parameters
        ----------
        hyperparameters : list of dict
            List of dictionaries containing hyperparameters. Each dict has two entries.
            ``"hyperparameters"`` contains a dictionary with the keyword arguments that
            must be passed to the celerite2 SHOTerm constructor. ``"metadata"`` is a
            dictionary describing which hyperparameters were fixed in the fit.
        name : str
            Label or name for the hyperparameter set.
        magnitude : float
            Magnitude of the target in the observational band.
        """
        super().__init__(hyperparameters)
        self.name = name
        self.magnitude = magnitude

    def __repr__(self):
        first = json.dumps(self[0], indent=4)
        return (
            f'<{self.__class__.__name__} ' +
            (f'"{self.name}" ' if self.name is not None else '') +
            f'(showing 1 of {len(self)}):\n[{first}...]>'
        )

    @staticmethod
    def _load_from_json(path):
        with open(path, 'r') as param_file:
            hyperparameters = json.load(param_file)
        return hyperparameters

    @classmethod
    def from_soho_virgo(cls, path=None, name='SOHO VIRGO/PMO6'):
        """
        Load the SOHO VIRGO/PMO6 total solar irradiance hyperparameters.

        See [1]_ for more information on SOHO VIRGO/PMO6.

        Parameters
        ----------
        path : str or None
            Path to the solar hyperparameter JSON file. If None, loads
            the default gadfly solar hyperparameter list.
        name : str
            Name of the hyperparameter set

        References
        ----------
        .. [1] `VIRGO Data Products Archive
           <https://www.pmodwrc.ch/en/research-development/solar-physics/virgo-data-products-archived_webpage/>`_
        """
        if path is None:
            path = default_hyperparameter_path
        hyperparameters = cls._load_from_json(path)

        return cls(hyperparameters, name=name)

    @classmethod
    @u.quantity_input(mass=u.g, temperature=u.K, radius=u.m, luminosity=u.L_sun)
    def for_star(
            cls, mass, radius, temperature, luminosity,
            bandpass=None, name=None, quiet=False, magnitude=None
    ):
        """
        Applying scaling relations to the SOHO VIRGO/PMO6 total solar
        irradiance hyperparameters for given stellar properties.

        Parameters
        ----------
        mass : ~astropy.units.Quantity
            Stellar mass
        radius : ~astropy.units.Quantity
            Stellar radius
        temperature : ~astropy.units.Quantity
            Effective temperature
        luminosity : ~astropy.units.Quantity
            Stellar luminosity
        bandpass : str
            Name of the observing bandpass.
        name : str
            Name for the star or set of hyperparameters
        quiet : bool
            Turn off warning for invalid input parameters to scaling
            relations when ``quiet=True``.
        magnitude : float
            Magnitude of the target star in the observing band
        """
        hyperparameters = cls._load_from_json(default_hyperparameter_path)

        # extract the granulation hyperparameters
        granulation_hyperparams = [
            item for item in hyperparameters
            if item['metadata']['source'] == 'granulation'
        ]

        # extract the p-mode hyperparameters
        p_mode_hyperparams = [
            item for item in
            sorted(
                hyperparameters,
                key=lambda x: x['metadata'].get('degree', -1)
            )
            if item['metadata']['source'] == 'oscillation'
        ]
        # put the p-mode hyperparams in a vector format, which is needed
        # shortly for the call to _p_mode_fit_to_sho_hyperparams
        p_mode_hyperparams = np.transpose(
            [[param_set['hyperparameters'].get(param)
              for param in ['S0', 'Q']]
             for param_set in p_mode_hyperparams]
        ).ravel()
        # get the hyperparameter sets and the corresponding labels to the
        # spherical degree "ell" for each set:
        (S0_fit, solar_w0, Q_fit), ell_labels = (
            _p_mode_fit_to_sho_hyperparams(p_mode_hyperparams)
        )

        solar_gran_S0, solar_gran_w0, solar_gran_Q = np.transpose(
            [[param_set['hyperparameters'].get(param)
              for param in ['S0', 'w0', 'Q']]
             for param_set in granulation_hyperparams]
        )

        # basic asteroseismic parameters:
        solar_nu_max = scale._solar_nu_max
        scaled_nu_max = solar_nu_max * scale.nu_max(mass, temperature, radius)

        # get access to the astronomical filter bandpass via tynt:
        filt = Filter(bandpass)

        # scale the hyperparameters for each of the granulation components
        scaled_hyperparameters = []
        for item in granulation_hyperparams:
            params = item['hyperparameters']

            scale_S0 = (
                params['S0'] *
                # scale the amplitudes by a term for granulation:
                scale.granulation_amplitude(
                    mass, temperature, luminosity
                ) *
                # also scale the amplitudes for the observing bandpass:
                scale.amplitude_with_wavelength(
                    filt, temperature
                )
            )

            # scale the timescales:
            scaled_w0 = params['w0'] / scale.tau_gran(
                mass, temperature, luminosity
            )

            if scaled_w0 > 0:
                scaled_hyperparameters.append(
                    dict(
                        hyperparameters=dict(
                            S0=scale_S0,
                            w0=scaled_w0,
                            Q=params['Q']),
                        metadata=item['metadata']
                    )
                )
            else:
                if not quiet:
                    msg = (
                        "The scaled solar hyperparameter with frequency "
                        f"w0(old)={params['w0']:.0f} is being scaled to "
                        f"w0(new)={scaled_w0:.0f}, which is not positive. "
                        f"This kernel term will be omitted."
                    )
                    warnings.warn(msg, AstropyUserWarning)

        # prepare to scale the hyperparameters for each of the
        # oscillation components, which also depend on the
        # amplitude of granulation at the frequency where the
        # oscillations occur:
        solar_nu = solar_w0 / (2 * np.pi) * u.uHz
        granulation_background_solar = _sho_psd(
            2 * np.pi * solar_nu.value[:, None],
            solar_gran_S0[None, :],
            solar_gran_w0[None, :], solar_gran_Q[None, :]
        )

        scale_delta_nu = scale.delta_nu(mass, radius)
        solar_delta_nu = solar_nu - solar_nu_max
        scaled_delta_nu = solar_delta_nu * scale_delta_nu
        scaled_nu = scaled_nu_max + scaled_delta_nu
        scaled_w0 = 2 * np.pi * scaled_nu.to(u.uHz).value

        # limit to positive scaled frequencies:
        # only_positive_omega = scaled_w0 > 0
        # solar_nu = solar_nu[only_positive_omega]
        # S0_fit = S0_fit[only_positive_omega]
        # Q_fit = Q_fit[only_positive_omega]
        # scaled_nu = scaled_nu[only_positive_omega]
        # scaled_w0 = scaled_w0[only_positive_omega]

        p_mode_scale_factor = (
            # scale p-mode "heights" like Kiefer et al. (2018)
            # as a function of frequency (scaled_nu)
            scale.p_mode_intensity(
                temperature, scaled_nu, scaled_nu_max,
                scale._solar_delta_nu * scale_delta_nu,
                filt.mean_wavelength
            ) *
            # scale the p-mode amplitudes like Kjeldsen & Bedding (2011)
            # according to stellar spectroscopic parameters:
            scale.p_mode_amplitudes(
                mass, radius, temperature, luminosity,
                scaled_nu, solar_nu,
                filt.mean_wavelength
            )
        )
        # scale the quality factors:
        scale_factor_Q = scale.quality(
            scaled_nu_max, temperature
        )
        scaled_Q = Q_fit * scale_factor_Q

        solar_psd_at_p_mode_peaks = _sho_psd(
            2 * np.pi * solar_nu.value, S0_fit,
            2 * np.pi * solar_nu.value, Q_fit
        ) * u.cds.ppm ** 2 / u.uHz

        # Following Chaplin 2008 Eqn 3:
        A = 2 * np.sqrt(4 * np.pi * solar_nu * solar_psd_at_p_mode_peaks)
        solar_mode_width = scale._lifetimes_lund(5777)
        scaled_mode_width = scale._lifetimes_lund(temperature.value)
        unscaled_height = 2 * A ** 2 / (np.pi * solar_mode_width)

        scaled_height = (
            unscaled_height *
            # scale to trace the envelope in power with frequency:
            p_mode_scale_factor *
            # scale to the correct bandpass:
            scale.amplitude_with_wavelength(
                filt, temperature
            )
        )
        scaled_A = np.sqrt(np.pi * scaled_mode_width * scaled_height / 2)
        scaled_psd_at_p_mode_peaks = (
            (scaled_A / 2)**2 / (4 * np.pi * scaled_nu)
        ).to(u.cds.ppm**2/u.uHz).value
        gran_background_solar = granulation_background_solar.sum(1)

        scaled_S0 = (
            np.pi * scaled_psd_at_p_mode_peaks /
            scaled_Q ** 2
        ) * gran_background_solar
        scaled_w0 = np.ravel(
            np.repeat(scaled_w0[None, :], len(S0_fit), 0)
        )
        scaled_Q = np.ravel(scaled_Q)

        for S0, w0, Q, degree in zip(scaled_S0, scaled_w0, scaled_Q, ell_labels):
            if np.all(np.array([S0, w0]) > 0):
                scaled_hyperparameters.append(
                    dict(
                        hyperparameters=dict(
                            S0=S0,
                            w0=w0,
                            Q=Q),
                        metadata=dict(
                            source='oscillation',
                            scaled=True,
                            degree=degree
                        )
                    )
                )

        return cls(scaled_hyperparameters, name, magnitude)


class StellarOscillatorKernel(celerite2_terms.TermConvolution):
    """
    A sum of :py:class:`~celerite2.terms.SHOTerm` simple harmonic oscillator
    kernels generated by gadfly to approximate the total solar irradiance
    power spectrum.

    :py:class:`~gadfly.core.StellarOscillatorKernel` inherits from
    :py:class:`~celerite2.terms.TermConvolution`.
    """
    @u.quantity_input(texp=u.s)
    def __init__(self, hyperparameters=None, texp=None, delta=None, name=None, terms=None):
        """
        Parameters
        ----------
        hyperparameters : ~gadfly.Hyperparameters
            Iterable of hyperparameters, each passed to the
            :py:class:`~celerite2.terms.SHOTerm` constructor.
        name : str
            Name for this set of hyperparameters
        texp : ~astropy.units.Quantity
            Exposure time, convertible to inverse microhertz.
        delta : float (optional)
            Exposure time, in units of inverse microhertz.
        terms : list
            Kernel terms to add together. This argument
            is intended only for internal use.
        """
        kernel_components = []

        if hyperparameters is not None:
            self.hyperparameters = hyperparameters

            if name is None and hyperparameters.name is not None:
                name = hyperparameters.name

            kernel_components += [
                celerite2_terms.SHOTerm(**p['hyperparameters']) for p in self.hyperparameters
            ]

        if terms is not None:
            kernel_components += terms

        self.name = name
        term_sum = celerite2_terms.TermSum(*kernel_components)

        if delta is None:
            if texp is None:
                default_exp = 1 * u.min
                msg = (
                    "An exposure time is required to construct the kernel. gadfly will assume "
                    f"a default exposure time of {default_exp}. To prevent this warning, supply "
                    f"the kernel with the `texp` keyword argument."
                )
                warnings.warn(msg, AstropyUserWarning)
                texp = default_exp

            delta = texp.to(1/u.uHz).value

        super().__init__(term_sum, delta)

    def plot(self, **kwargs):
        return plot_power_spectrum(kernel=self, **kwargs)

    plot.__doc__ = plot_power_spectrum.__doc__

    @classmethod
    def _from_terms(cls, terms, delta=None, name=None):
        return cls(terms=terms, delta=delta, name=name)

    def __add__(self, other):
        """
        Assumes ``other`` is a SHOTerm or subclass.
        """
        if not isinstance(other, list):
            other_names = [other.name]
            other = [other]
        else:
            other_names = [t.name for t in other.terms]

        name = ""
        if self.name is not None:
            name += self.name
        for other_name in other_names:
            if other_name is not None:
                if len(name):
                    name += " + " + other_name
                else:
                    name += other_name

        return StellarOscillatorKernel._from_terms(
            list(self.term.terms) + other, delta=self.delta, name=name
        )


class SolarOscillatorKernel(StellarOscillatorKernel):
    """
    Like a :py:class:`~gadfly.core.StellarOscillatorKernel`, but initialized
    with the default solar SOHO VIRGO/PMO6 kernel hyperparameters. The hyperparameters
    are initialized with the :py:class:`~gadfly.core.Hyperparameters` class method
    :py:meth:`~gadfly.core.Hyperparameters.for_star` assuming exactly
    solar mass, radius, temperature, and luminosity.

    The primary difference with :py:class:`~gadfly.core.StellarOscillatorKernel` is
    that the user need not provide :py:meth:`~gadfly.core.Hyperparameters`.

    :py:class:`~gadfly.core.SolarOscillatorKernel` inherits from
    :py:class:`~celerite2.terms.TermConvolution` and
    :py:class:`~gadfly.core.StellarOscillatorKernel`.
    """

    @u.quantity_input(texp=u.s)
    def __init__(self, texp=None, delta=None, bandpass=None, name=None):
        """
        Parameters
        ----------
        texp : ~astropy.units.Quantity
            Exposure time, convertible to inverse microhertz.
        """
        hp = Hyperparameters.for_star(
            mass=1*u.M_sun, radius=1*u.R_sun,
            temperature=5777*u.K, luminosity=1*u.L_sun,
            bandpass=bandpass
        )
        super().__init__(
            hp, texp=texp, delta=delta, name=name
        )


class ShotNoiseKernel(celerite2_terms.SHOTerm):
    """
    A subclass of :py:class:`~celerite2.terms.SHOTerm` which approximates
    shot noise in Kepler observations, for example.
    """

    # This is intentionally really large. In [uHz].
    w0 = 1e7

    # value does not matter much if w0 >>> 1
    Q = 0.5

    def __init__(self, *args, name=None, **kwargs):
        """
        Parameters
        ----------
        args : dict
            Hyperparameters for the SHO kernel, :math:`S_0, \\omega_0, Q`.
        name : str
            Name to store for the target/instrument.
        kwargs : dict
            Extra keyword arguments to pass to the
            :py:class:`~celerite2.terms.SHOTerm` constructor.
        """
        if name is None:
            name = "Shot noise"
        self.name = name
        super().__init__(*args, **kwargs)

    @classmethod
    def from_kepler_light_curve(cls, light_curve):
        """
        Estimate Kepler shot noise from the light curve.

        The light curve must have a Kepler magnitude in the metadata.
        Assumes the noise relation in Jenkins et al. (2010) [1]_.

        Parameters
        ----------
        light_curve : ~lightkurve.lightcurve.LightCurve
            Kepler light curve.

        References
        ----------
        .. [1] `Jenkins et al. (2010)
           <https://ui.adsabs.harvard.edu/abs/2010ApJ...713L.120J/abstract>`_
        """
        from lightkurve import LightCurveCollection

        if isinstance(light_curve, LightCurveCollection):
            light_curve = light_curve.stitch(lambda x: x)

        kepler_mag = light_curve.meta['KEPMAG']
        norm = 2 * np.pi / len(light_curve.time) ** 0.5
        _, unscaled_S0 = cls.kepler_mag_to_noise_amplitude(kepler_mag)
        S0 = (unscaled_S0 * norm) ** 0.5
        return cls(S0=S0.to(u.cds.ppm).value, w0=cls.w0, Q=cls.Q)

    @staticmethod
    def kepler_mag_to_noise_amplitude(kepler_mag):
        """
        Kepler noise in 6 hour bins, from Jenkins et al. (2010) [1]_.

        Parameters
        ----------
        kepler_mag : float
            Kepler magnitude (:math:`K_p`) of the star.

        References
        ----------
        .. [1] `Jenkins et al. (2010)
           <https://ui.adsabs.harvard.edu/abs/2010ApJ...713L.120J/abstract>`_
        """
        c = 3.46 * 10 ** (0.4 * (12 - kepler_mag) + 8)
        sigma_lower = np.sqrt(
            c + 7e6 * np.max([np.ones_like(kepler_mag), kepler_mag / 14], axis=0) ** 4
        ) / c  # units of relative flux
        sigma_upper = np.sqrt(c + 7e7) / c

        # convert to ppm:
        return 1e6 * np.array([sigma_lower, sigma_upper]) * u.cds.ppm ** 2


class Filter(tynt.Filter):
    """
    Convenience subclass for the ``tynt`` API for photometric
    filter transmittance curves.

    ``tynt`` is a parameterized filter bandpass package [1]_.

    References
    ----------
    .. [1] `tynt source code on GitHub <https://github.com/bmorris3/tynt>`_.
    """
    generator = tynt.FilterGenerator()
    available_filters = generator.available_filters()
    default_filter = 'Kepler/Kepler.K'

    def __init__(self, identifier_or_filter, download=False):
        """
        Parameters
        ----------
        identifier_or_filter : str, ~gadfly.Filter, ~tynt.Filter
            Input name of the filter, or a ``gadfly`` or ``tynt``
            implementation of the filter object.
        download : bool
            If True, download the true transmittance curve. Default
            is False.
        """
        # Use Kepler bandpass if None specified, and give a warning.
        if identifier_or_filter is None:
            msg = (
                "An observing bandpass is required to construct the kernel. gadfly "
                f'will assume the default filter "{self.default_filter}". To prevent '
                f"this warning, supply the Hyperparameters with the `bandpass` "
                "keyword argument."
            )
            warnings.warn(msg, AstropyUserWarning)
            identifier_or_filter = self.default_filter

        # Define a "bandpass" for SOHO, which is bolometric
        if identifier_or_filter.upper() == 'SOHO VIRGO':
            wavelength = np.logspace(-1.5, 1.5, 1000) * u.um
            super().__init__(wavelength, np.ones_like(wavelength.value))

        else:
            if isinstance(identifier_or_filter, (tynt.Filter, Filter)):
                # this happens if the user supplies a filter, we just
                # return the same filter:
                return identifier_or_filter
            else:
                # otherwise try to reconstruct the bandpass transmittance from
                # the tynt FFT parameterization:
                if identifier_or_filter in self.available_filters and not download:
                    filt = self.generator.reconstruct(identifier_or_filter)
                elif not download:
                    msg = (
                        f'The observing bandpass "{identifier_or_filter}" is not recognized '
                        f'in the pre-loaded bandpasses in tynt, and the `download` keyword is '
                        f'"{download}". To trigger a remote filter bandpass download from the '
                        f'SVO FPS, set `download=True`.'
                    )
                    raise ValueError(msg)
                else:
                    filt = self.generator.download_true_transmittance(identifier_or_filter)
                super().__init__(filt.wavelength, filt.transmittance)

    @property
    def mean_wavelength(self):
        """
        Transmittance-weighted mean wavelength of the filter bandpass.
        """
        return np.average(self.wavelength, weights=self.transmittance)
