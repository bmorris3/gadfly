import pytest
import numpy as np
import astropy.units as u
from ..sun import broomhall_p_mode_freqs

example_freqs = u.Quantity([
    3033.886, 3082.471, 3098.327, 3160.028, 3168.773, 3217.916
], u.uHz)


@pytest.mark.parametrize("example_freq,", example_freqs)
def test_broomhall(example_freq):
    freqs = broomhall_p_mode_freqs()

    assert isinstance(freqs, u.Quantity)

    closest_freq_in_table = freqs[np.argmin(np.abs(freqs - example_freq))]
    np.testing.assert_allclose(example_freq, closest_freq_in_table)
