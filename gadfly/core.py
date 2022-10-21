import os
import json

import numpy as np
from celerite2 import terms
import astropy.units as u

from . import scale
from .psd import plot_power_spectrum

__all__ = [
    'Hyperparameters', 'StellarOscillatorKernel', 'SolarOscillatorKernel'
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
    def __init__(self, hyperparameters, scale_factors=None, name=None):
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
        """
        super().__init__(hyperparameters)
        self.scale_factors = scale_factors
        self.name = name

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
    def for_star(cls, mass, radius, temperature, luminosity, name=None, quiet=False):
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

                scaled_hyperparameters.append(
                    dict(
                        hyperparameters=dict(
                            S0=scale_S0,
                            w0=scaled_w0,
                            Q=params['Q']),
                        metadata=item['metadata']
                    )
                )

            # scale the hyperparameters for the p-mode oscillations:
            else:
                params = item['hyperparameters']

                # scale the p-mode amplitudes, along with the
                # granulation scaling:
                scaled_S0 = (
                    params['S0'] * scale_factors['p_mode_amps'] *
                    scale.granulation_amplitude(mass, temperature, luminosity)
                )

                # scale the p-mode frequencies
                solar_delta_nu = (
                    params['w0'] / (2 * np.pi) * u.uHz -
                    scale._solar_nu_max
                )
                scaled_delta_nu = solar_delta_nu * scale_factors['delta_nu']
                scaled_w0 = 2 * np.pi * (scaled_nu_max + scaled_delta_nu).to(u.uHz).value

                # scale the quality factors. The FWHM is proportional to the inverse of the
                # Q factor, so we divide by the scale factor instead of multiplying it.
                scaled_Q = params['Q'] / scale_factors['fwhm']

                scaled_hyperparameters.append(
                    dict(
                        hyperparameters=dict(
                            S0=scaled_S0,
                            w0=scaled_w0,
                            Q=scaled_Q),
                        metadata=item['metadata']
                    )
                )

        return cls(scaled_hyperparameters, scale_factors, name)


class StellarOscillatorKernel(terms.TermSum):
    """
    A sum of :py:class:`~celerite2.terms.SHOTerm` simple harmonic oscillator
    kernels generated by gadfly to approximate the total solar irradiance
    power spectrum.

    :py:class:`~gadfly.core.StellarOscillatorKernel` inherits from
    :py:class:`~celerite2.terms.TermSum`.
    """

    def __init__(self, hyperparameters, name=None):
        """
        Parameters
        ----------
        hyperparameters : ~gadfly.Hyperparameters
            Iterable of hyperparameters, each passed to the
            :py:class:`~celerite2.terms.SHOTerm` constructor.
        name : str
            Name for this set of hyperparameters
        """
        self.hyperparameters = hyperparameters
        self.name = name

        kernel_components = [
            terms.SHOTerm(**p['hyperparameters']) for p in self.hyperparameters
        ]

        super().__init__(*kernel_components)

    def plot(self, **kwargs):
        """
        See docstring for :py:func:`~gadfly.plot_power_spectrum` for arguments
        """
        return plot_power_spectrum(kernel=self, **kwargs)


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
