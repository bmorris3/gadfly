import os
import json

import numpy as np
from celerite2 import terms
import astropy.units as u

from gadfly import scale

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
    def __init__(self, hyperparameters, scale_factors=None):
        """
        Parameters
        ----------
        hyperparameters : list of dict
            List of dictionaries containing hyperparameters. Each dict has two entries.
            `"hyperparameters"` contains a dictionary with the keyword arguments that
            must be passed to the celerite2 SHOTerm constructor. `"metadata"` is a
            dictionary describing which hyperparameters were fixed in the fit.
        scale_factors : dict
            Scaling relation scale-factors to apply to solar hyperparameters.
        """
        # strip off metadata
        super().__init__(hyperparameters)
        self.scale_factors = scale_factors

    def __repr__(self):
        first = json.dumps(self[0], indent=2)
        return f"<{self.__class__.__name__} (showing 1 of {len(self)}):\n[{first}...]>"

    @staticmethod
    def _load_from_json(path):
        with open(path, 'r') as param_file:
            hyperparameters = json.load(param_file)
        return hyperparameters

    @classmethod
    def from_soho_virgo(cls, path=None):
        """
        Load the SOHO VIRGO/PMO6 total solar irradiance hyperparameters.

        Parameters
        ----------
        path : str or None
            Path to the solar hyperparameter JSON file. If None, loads
            the default gadfly solar hyperparameter list.
        """
        if path is None:
            path = default_hyperparameter_path
        hyperparameters = cls._load_from_json(path)

        return cls(hyperparameters)

    @staticmethod
    @u.quantity_input(mass=u.g, temperature=u.K, radius=u.m, luminosity=u.L_sun)
    def _get_scale_factors(mass, temperature, radius, luminosity, quiet=False):
        return dict(
            p_mode_amps=scale.p_mode_amplitudes(mass, temperature, luminosity),
            nu_max=scale.nu_max(mass, temperature, radius),
            delta_nu=scale.delta_nu(mass, radius),
            fwhm=scale.fwhm(temperature, quiet=quiet)
        )

    @classmethod
    @u.quantity_input(mass=u.g, temperature=u.K, radius=u.m, luminosity=u.L_sun)
    def for_star(cls, mass, temperature, radius, luminosity, quiet=False):
        """
        Applying scaling relations to the SOHO VIRGO/PMO6 total solar
        irradiance hyperparameters for given stellar properties.

        Parameters
        ----------
        mass : ~astropy.units.Quantity
            Stellar mass
        temperature : ~astropy.units.Quantity
            Effective temperature
        radius : ~astropy.units.Quantity
            Stellar radius
        luminosity : ~astropy.units.Quantity
            Stellar luminosity
        """
        hyperparameters = cls._load_from_json(default_hyperparameter_path)
        scale_factors = cls._get_scale_factors(
            mass, temperature, radius, luminosity, quiet=quiet
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
                    params['w0'] / (2*np.pi) * u.uHz, scaled_nu_max,
                    mass, radius, temperature, luminosity
                )

                # scale the timescales:
                scaled_w0 = params['w0'] / scale.tau_gran(scaled_nu_max)

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

                # scale the p-mode amplitudes
                scaled_S0 = params['S0'] * scale_factors['p_mode_amps']

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

        return cls(scaled_hyperparameters, scale_factors)


class StellarOscillatorKernel(terms.TermSum):
    """
    A sum of simple harmonic oscillator kernels generated by gadfly to
    approximate the total solar irradiance power spectrum.
    """

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

        kernel_components = [
            terms.SHOTerm(**p['hyperparameters']) for p in self.hyperparameters
        ]

        super().__init__(*kernel_components)


class SolarOscillatorKernel(StellarOscillatorKernel):
    """
    Like a ``StellarOscillatorKernel``, but limited to only
    the default gadfly SOHO VIRGO/PMO6 kernel hyperparameters.
    """

    def __init__(self):
        super().__init__(
            Hyperparameters.from_soho_virgo()
        )
