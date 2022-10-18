import pytest
import numpy as np

import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning

from ..core import Hyperparameters
from ..scale import (
    fwhm, _solar_mass, _solar_temperature, _solar_radius, _solar_luminosity
)


def test_scaling_relations():
    # initialize a "star" with exactly solar parameters:
    params = Hyperparameters.for_star(
        _solar_mass, _solar_radius, _solar_temperature, _solar_luminosity
    )

    # now compute scale factors and ensure they're all unity
    should_be_ones = np.array(list(params.scale_factors.values()))
    np.testing.assert_allclose(np.ones_like(should_be_ones), should_be_ones)


def test_process_inputs():
    with pytest.warns(AstropyUserWarning):
        fwhm(3500*u.K)
