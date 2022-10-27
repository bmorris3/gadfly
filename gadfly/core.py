import os
import json
import warnings

import numpy as np

from celerite2 import terms as celerite2_terms

import astropy.units as u
from astropy.units import cds  # noqa
from astropy.utils.exceptions import AstropyUserWarning

from . import scale
from .psd import plot_power_spectrum

__all__ = [
    'Hyperparameters',
    'StellarOscillatorKernel',
    'SolarOscillatorKernel',
    'ShotNoiseKernel'
]

dirname = os.path.dirname(os.path.abspath(__file__))
default_hyperparameter_path = os.path.join(
    dirname, 'data', 'hyperparameters.json'
)


class Hyperparameters(list):
    """
    Gaussian process hyperparameters for approximating
    the total stellar irradiance power spectrum.
    """
    def __init__(self, hyperparameters, scale_factors=None, name=None, magnitude=None):
        """
        Parameters
        ----------
        hyperparameters : list of dict
            List of dictionaries containing hyperparameters. Each dict has two entries.
            `"hyperparameters"` contains a dictionary with the keyword arguments that
            must be passed to the celerite2 SHOTerm constructor. `"metadata"` is a
            dictionary describing which hyperparameters were fixed in the fit.
        scale_factors : dict or None
            Scaling relation scale-factors to apply to solar hyperparameters.
        name : str
            Label or name for the hyperparameter set.
        magnitude : float
            Magnitude of the target in the observational band.
        """
        super().__init__(hyperparameters)
        self.scale_factors = scale_factors
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

    @staticmethod
    @u.quantity_input(mass=u.g, temperature=u.K, radius=u.m, luminosity=u.L_sun)
    def _get_scale_factors(mass, radius, temperature, luminosity, quiet=False):
        return dict(
            p_mode_amps=scale.p_mode_amplitudes(mass, temperature, luminosity),
            nu_max=scale.nu_max(mass, temperature, radius),
            delta_nu=scale.delta_nu(mass, radius),
            fwhm=scale.fwhm(temperature, quiet=quiet)
        )

    @classmethod
    @u.quantity_input(mass=u.g, temperature=u.K, radius=u.m, luminosity=u.L_sun)
    def for_star(
            cls, mass, radius, temperature, luminosity,
            name=None, quiet=False, magnitude=None):
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
        name : str
            Name for the star or set of hyperparameters
        quiet : bool
            Turn off warning for invalid input parameters to scaling
            relations when ``quiet=True``.
        magnitude : float
            Magnitude of the target star in the observing band
        """
        hyperparameters = cls._load_from_json(default_hyperparameter_path)
        scale_factors = cls._get_scale_factors(
            mass, radius, temperature, luminosity, quiet=quiet
        )
        scaled_nu_max = scale._solar_nu_max * scale_factors['nu_max']

        scaled_hyperparameters = []
        for item in hyperparameters:
            is_fixed_Q = 'Q' in item['metadata']['fixed_parameters']

            # scale the hyperparameters for the low-frequency features:
            if is_fixed_Q:
                params = item['hyperparameters']

                # scale the granulation amplitudes
                scale_S0 = params['S0'] * scale.granulation_amplitude(
                    mass, temperature, luminosity
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

            # scale the hyperparameters for the p-mode oscillations:
            else:
                params = item['hyperparameters']

                # scale the p-mode amplitudes
                scaled_S0 = (
                    params['S0'] * scale_factors['p_mode_amps'] * scale.granulation_amplitude(
                        mass, temperature, luminosity
                    )
                )

                # scale the p-mode frequencies
                solar_delta_nu = (
                    params['w0'] / (2 * np.pi) * u.uHz - scale._solar_nu_max
                )
                scaled_delta_nu = solar_delta_nu * scale_factors['delta_nu']
                scaled_w0 = 2 * np.pi * (scaled_nu_max + scaled_delta_nu).to(u.uHz).value

                # scale the quality factors by scaling the p-mode peaks' FWHM:
                unscaled_fwhm = scaled_w0 / (2 * params['Q'])  # this goes like tau^-1
                scaled_fwhm = unscaled_fwhm * scale_factors['fwhm']
                scaled_Q = scaled_w0 / (2 * scaled_fwhm)

                if scaled_w0 > 0:
                    scaled_hyperparameters.append(
                        dict(
                            hyperparameters=dict(
                                S0=scaled_S0,
                                w0=scaled_w0,
                                Q=scaled_Q),
                            metadata=item['metadata']
                        )
                    )
                else:
                    if not quiet:
                        msg = (
                            "The scaled solar (p-mode) hyperparameter with frequency "
                            f"w0(old)={params['w0']:.0f} is being scaled to "
                            f"w0(new)={scaled_w0:.0f}, which is not positive. "
                            f"This kernel term will be omitted."
                        )
                        warnings.warn(msg, AstropyUserWarning)

        return cls(scaled_hyperparameters, scale_factors, name, magnitude)


class StellarOscillatorKernel(celerite2_terms.TermSum):
    """
    A sum of :py:class:`~celerite2.terms.SHOTerm` simple harmonic oscillator
    kernels generated by gadfly to approximate the total solar irradiance
    power spectrum.

    :py:class:`~gadfly.core.StellarOscillatorKernel` inherits from
    :py:class:`~celerite2.terms.TermSum`.
    """

    def __init__(self, hyperparameters=None, name=None, terms=None):
        """
        Parameters
        ----------
        hyperparameters : ~gadfly.Hyperparameters
            Iterable of hyperparameters, each passed to the
            :py:class:`~celerite2.terms.SHOTerm` constructor.
        name : str
            Name for this set of hyperparameters
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

        super().__init__(*kernel_components)

    def plot(self, **kwargs):
        """
        See docstring for :py:func:`~gadfly.plot_power_spectrum` for arguments
        """
        return plot_power_spectrum(kernel=self, **kwargs)

    @classmethod
    def _from_terms(cls, terms, name=None):
        return cls(terms=terms, name=name)

    def __add__(self, other):
        """
        Assumes ``other`` is a SHOTerm or subclass. Adds that kernel to the
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
            tuple(list(self.terms) + other), name=name
        )


class SolarOscillatorKernel(StellarOscillatorKernel):
    """
    Like a :py:class:`~gadfly.core.StellarOscillatorKernel`, but initialized
    with the default gadfly SOHO VIRGO/PMO6 kernel hyperparameters. These parameters
    are initialized with the :py:class:`~gadfly.core.Hyperparameters` class method
    :py:meth:`~gadfly.core.Hyperparameters.from_soho_virgo`.

    :py:class:`~gadfly.core.SolarOscillatorKernel` inherits from
    :py:class:`~celerite2.terms.TermSum` and
    :py:class:`~gadfly.core.StellarOscillatorKernel`.
    """

    def __init__(self):
        hp = Hyperparameters.from_soho_virgo()
        super().__init__(
            hp, name=hp.name
        )


class ShotNoiseKernel(celerite2_terms.SHOTerm):
    """
    A subclass of :py:class:`~celerite2.terms.SHOTerm` which approximates
    shot noise in Kepler observations, for example.
    """

    w0 = 1e10  # in [uHz]. This is intentionally really large.
    Q = 0.5  # value does not matter much if w0 >>> 1

    def __init__(self, *args, name=None, **kwargs):
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
