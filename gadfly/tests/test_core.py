import numpy as np

import astropy.units as u
from astropy.units import cds  # noqa

from lightkurve import LightCurve

from ..core import SolarOscillatorKernel
from ..gp import GaussianProcess
from ..psd import PowerSpectrum
from ..scale import (
    _solar_mass, _solar_temperature, _solar_radius,
    amplitude_with_wavelength, nu_max, delta_nu
)


def test_ps_lc_round_trip(n_trials=10, bin=15):
    # test round trip from PowerSpectrum -> LightCurve -> PowerSpectrum.bin()
    np.random.seed(42)
    power_units = u.cds.ppm ** 2 / u.uHz

    kernel = SolarOscillatorKernel(texp=1 * u.min, bandpass='SOHO VIRGO')

    t = np.linspace(0, 100, int(1e5)) * u.d

    gp = GaussianProcess(kernel, t=t)

    for i in range(n_trials):
        synth_flux = gp.sample(return_quantity=True)
        synth_lc = LightCurve(time=t, flux=synth_flux)
        synth_ps = PowerSpectrum.from_light_curve(synth_lc).bin(bin)
        synth_power = synth_ps.power.to(power_units).value
        synth_power_err = synth_ps.error.to(power_units).value
        kernel_power = kernel.get_psd(2 * np.pi * synth_ps.frequency.to(u.uHz).value)

        # put an upper and lower freq limit on the comparison:
        skip_p_modes = synth_ps.frequency < 1e3 * u.uHz
        skip_lowest_freq = synth_ps.frequency > 3 * u.uHz

        in_bounds = skip_p_modes & skip_lowest_freq

        # compute difference between model input and simulated output
        # in units of sigma (err):
        deviation = (
            np.abs((kernel_power[in_bounds] - synth_power[in_bounds]) /
                   synth_power_err.max())
        )
        # pass if the maximum deviation is <5 sigma. Skip nans (empty bins):
        assert np.nanmax(deviation) < 5


def test_scaling_relations_to_solar():
    bandpass = 'SOHO VIRGO'
    # initialize a "star" with exactly solar parameters:
    mass, rad, temp = (
        _solar_mass, _solar_radius, _solar_temperature
    )

    # now compute scale factors and ensure they're all unity
    should_be_ones = np.array([
        amplitude_with_wavelength(bandpass, temp),
        nu_max(mass, temp, rad),
        delta_nu(mass, rad)
    ])
    np.testing.assert_allclose(np.ones_like(should_be_ones), should_be_ones)


def test_scale_amp_with_wavelength():
    assert amplitude_with_wavelength('SOHO VIRGO', _solar_temperature) == 1
    kepler_amp = amplitude_with_wavelength('Kepler/Kepler.K', _solar_temperature)
    np.testing.assert_allclose(
        kepler_amp, 2.795, rtol=1e-3
    )
