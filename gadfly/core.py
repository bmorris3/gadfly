import numpy as np
from celerite2 import terms, GaussianProcess
import astropy.units as u
from astropy.constants import L_sun, M_sun, R_sun
import os

__all__ = ['generate_solar_fluxes', 'generate_stellar_fluxes']

dirname = os.path.dirname(os.path.abspath(__file__))
PARAM_VECTOR = np.loadtxt(
    os.path.join(dirname, 'data', 'parameter_vector.txt')
)


def _process_inputs(duration, cadence, T_eff):
    """
    Check for sensible inputs.

    Raises
    ------
    ValueError
        If duration is less than or equal to the cadence.
    """
    if duration <= cadence:
        raise ValueError("``duration`` must be longer than ``cadence``")
    if T_eff is not None and T_eff < 4900 * u.K:
        raise ValueError("Only valid for temperatures >4900 K.")


@u.quantity_input(cadence=u.s, duration=u.s)
def generate_solar_fluxes(duration, cadence=60 * u.s, seed=None):
    """
    Generate an array of fluxes with zero mean which mimic the power spectrum of
    the SOHO/VIRGO SPM observations.

    Parameters
    ----------
    duration : `~astropy.units.Quantity`
        Duration of simulated observations to generate.
    cadence : `~astropy.units.Quantity`
        Length of time between fluxes
    seed : float, optional
        Random seed.

    Returns
    -------
    times : `~astropy.units.Quantity`
        Array of times at cadence ``cadence`` of length ``duration/cadence``
    fluxes : `~numpy.ndarray`
        Array of fluxes at cadence ``cadence`` of length ``duration/cadence``
    kernel : `~celerite2.terms.TermSum`
        Celerite kernel used to approximate the solar power spectrum.
    """
    if seed is not None:
        np.random.seed(seed)

    _process_inputs(duration, cadence, 5777 * u.K)

    ##########################
    # Assemble celerite kernel
    ##########################
    parameter_vector = np.copy(PARAM_VECTOR)
    nterms = len(parameter_vector) // 3
    params = parameter_vector.reshape((nterms, 3))

    nterms = len(parameter_vector) // 3

    kernel = (
        # Granulation terms
        terms.SHOTerm(S0=params[0, 0] * 1e-6, w0=1e-6 * params[0, 1],
                      Q=params[0, 2]) +
        terms.SHOTerm(S0=params[1, 0] * 1e-6, w0=1e-6 * params[1, 1],
                      Q=params[1, 2])
    )

    for term in range(2, nterms):
        # p-mode terms
        kernel += terms.SHOTerm(S0=params[term, 0] * 1e-6, w0=params[term, 1],
                                Q=params[term, 2])

    gp = GaussianProcess(kernel)

    times = np.arange(0, duration.to(u.s).value, cadence.to(u.s).value) * u.s
    x = times.value

    gp.compute(x, check_sorted=False)

    ###################################
    # Get samples with the kernel's PSD
    ###################################

    y = gp.sample()
    # Remove a linear trend:
    y -= np.polyval(np.polyfit(x - x.mean(), y, 1), x - x.mean())

    return times, y, kernel


@u.quantity_input(duration=u.s, cadence=u.s, M=u.kg, T_eff=u.K, L=u.W, R=u.m)
def generate_stellar_fluxes(duration, M, T_eff, R, L, cadence=60 * u.s,
                            frequencies=None, amplitudes=None,
                            mode_lifetimes=None, seed=None):
    """
    Generate an array of fluxes with zero mean which mimic the power spectrum of
    the SOHO/VIRGO SPM observations, scaled for a star with a given mass,
    effective temperature, luminosity and radius.

    Parameters
    ----------
    duration : `~astropy.units.Quantity`
        Duration of simulated observations to generate.
    M : `~astropy.units.Quantity`
        Stellar mass
    T_eff : `~astropy.units.Quantity`
        Stellar effective temperature
    R : `~astropy.units.Quantity`
        Stellar radius
    L : `~astropy.units.Quantity`
        Stellar luminosity
    cadence : `~astropy.units.Quantity`
        Length of time between fluxes
    frequencies : `~numpy.ndarray` or None
        p-mode frequencies in the power spectrum in units of microHertz.
        Defaults to scaled solar values.
    amplitudes : `~numpy.ndarray` or None
        p-mode amplitudes in the power spectrum. Defaults to scaled solar
        values.
    mode_lifetimes : `~numpy.ndarray` or None
        p-mode lifetimes in the power spectrum. Defaults to scaled solar
        values.
    seed : float, optional
        Random seed.

    Returns
    -------
    times : `~astropy.units.Quantity`
        Array of times at cadence ``cadence`` of size ``duration/cadence``
    fluxes : `~numpy.ndarray`
        Array of fluxes at cadence ``cadence`` of size ``duration/cadence``
    kernel : `~celerite2.terms.TermSum`
        Celerite kernel used to approximate the stellar power spectrum
    """
    if seed is not None:
        np.random.seed(seed)

    _process_inputs(duration, cadence, T_eff)

    if frequencies is None:

        ##########################
        # Scale p-mode frequencies
        ##########################
        parameter_vector = np.copy(PARAM_VECTOR)

        # Scale frequencies

        tunable_amps = parameter_vector[::3][2:]
        tunable_freqs = parameter_vector[2::3][2:] * 1e6 / 2 / np.pi
        peak_ind = np.argmax(tunable_amps)
        peak_freq = tunable_freqs[peak_ind]  # 3090 uHz in Huber 2011
        delta_freqs = tunable_freqs - peak_freq

        T_eff_solar = 5777 * u.K

        # Huber 2012 Eqn 3
        delta_nu_factor = (M / M_sun) ** 0.5 * (R / R_sun) ** (-3 / 2)
        # Huber 2012 Eqn 4
        nu_factor = (
            (M / M_sun) * (R / R_sun) ** -2 * (T_eff / T_eff_solar) ** -0.5
        )

        new_peak_freq = nu_factor * peak_freq
        new_delta_freqs = delta_freqs * delta_nu_factor

        new_freqs = new_peak_freq + new_delta_freqs

        new_omegas = 2 * np.pi * new_freqs * 1e-6

        parameter_vector[2::3][2:] = new_omegas

        #############################################
        # Scale mode lifetimes of p-mode oscillations
        #############################################

        q = parameter_vector[1::3][2:]
        fwhm = 1 / (2 * np.pi * q)

        # From Enrico Corsaro (private communication), see Figure 7 of Corsaro 2015,
        # where X is T_eff.
        def ln_FWHM(X):
            return (1463.49 - 1.03503 * X + 0.000271565 * X ** 2 -
                    3.14139e-08 * X ** 3 + 1.35524e-12 * X ** 4)

        fwhm_scale = (
            np.exp(ln_FWHM(5777)) /
            np.exp(ln_FWHM(np.max([T_eff.value, 4900])))
        )

        scaled_fwhm = fwhm * fwhm_scale
        scaled_q = 1 / (2 * np.pi * scaled_fwhm)
        parameter_vector[1::3][2:] = scaled_q

        ##############################################################
        # Scale amplitudes of p-mode oscillations following Huber 2011
        ##############################################################

        # Huber 2011 Eqn 8:
        c = (T_eff / (5934 * u.K)) ** 0.8
        c_sun = ((5777 * u.K) / (5934 * u.K)) ** 0.8
        r = 2
        s = 0.886
        t = 1.89

        # Huber 2011 Eqn 9:
        pmode_amp_star = (float(L / L_sun) ** s /
                          (float(M / M_sun) ** t * T_eff.value ** (r - 1) * c))
        pmode_amp_sun = ((L_sun / L_sun) ** s /
                         ((M_sun / M_sun) ** t * 5777 ** (r - 1) * c_sun))
        pmode_amp_factor = pmode_amp_star / pmode_amp_sun

        new_pmode_amps = parameter_vector[0::3][2:] * pmode_amp_factor

        parameter_vector[0::3][2:] = new_pmode_amps

        #############################
        # Scale granulation frequency
        #############################

        # Kallinger 2014 pg 12:
        tau_eff_factor = (new_peak_freq / peak_freq) ** -0.89
        parameter_vector[2] = parameter_vector[2] / tau_eff_factor
        parameter_vector[5] = parameter_vector[5] / tau_eff_factor
        # Kjeldsen & Bedding (2011):
        granulation_amplitude_factor = (new_peak_freq / peak_freq) ** -2
        parameter_vector[0] = (parameter_vector[0] *
                               granulation_amplitude_factor)
        parameter_vector[3] = (parameter_vector[3] *
                               granulation_amplitude_factor)
        print(parameter_vector)
    else:

        omegas = 2 * np.pi * frequencies * 1e-6
        custom_params = np.vstack([amplitudes, mode_lifetimes,
                                   omegas]).T.ravel()
        parameter_vector = np.concatenate([PARAM_VECTOR[:6], custom_params])

    ##########################
    # Assemble celerite kernel
    ##########################
    params = parameter_vector.reshape((-1, 3))
    nterms = len(parameter_vector) // 3

    kernel = (
        # Granulation terms
        terms.SHOTerm(S0=params[0, 0] * 1e-6,
                      w0=1e-6 * params[0, 1],
                      Q=params[0, 2]) +
        terms.SHOTerm(S0=params[1, 0] * 1e-6,
                      w0=1e-6 * params[1, 1],
                      Q=params[1, 2])
    )

    for term in range(2, nterms):
        # p-mode terms
        kernel += terms.SHOTerm(S0=params[term, 0] * 1e-6,
                                w0=params[term, 1],
                                Q=params[term, 2])

    gp = GaussianProcess(kernel)

    times = np.arange(0, duration.to(u.s).value, cadence.to(u.s).value) * u.s
    x = times.value

    gp.compute(x, check_sorted=False)

    ###################################
    # Get samples with the kernel's PSD
    ###################################

    y = gp.sample()
    # Remove a linear trend
    y -= np.polyval(np.polyfit(x - x.mean(), y, 1), x - x.mean())

    return times, y, kernel
