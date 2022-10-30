Validation with Kepler
======================

Outline
-------

All of those scaling relations in :py:mod:`~gadfly.scale`
sound great, but do they yield kernels that accurately
reproduce Kepler time series photometric observations of
stars? Let's find out!

In :doc:`start`, we wrote out the
spectroscopically-derived stellar parameters for a handful of
stars. In this example, we'll use
`lightkurve <https://docs.lightkurve.org/>`_ to download the
Kepler observations for each of the stars. Then we'll use
``gadfly`` to do the following:

1. Load the ``celerite2`` kernel :py:class:`~gadfly.Hyperparameters`
   for the Sun, based on the SOHO VIRGO/PMO6 one-minute cadence
   observations
2. Scale those kernel hyperparameters with asteroseismic scaling
   relations for a star with known properties like mass, radius,
   temperature, and luminosity, with the
   :py:meth:`~gadfly.Hyperparameters.for_star` method.
3. Download the Kepler photometry for each target into a
   :py:class:`~lightkurve.LightCurve` object, with help from
   :py:func:`~lightkurve.search_lightcurve`.
4. Compute a :py:class:`~gadfly.PowerSpectrum` from the
   with the :py:meth:`~gadfly.PowerSpectrum.from_light_curve` method.
5. Plot the results for each star with the help of the
   :py:meth:`~gadfly.PowerSpectrum.plot` method.

Implementation
--------------

First, let's import the required pacakges, enumerate the stellar
parameters that we'll need later, and initialize the plot:

.. code-block:: python

    import warnings
    import numpy as np
    import matplotlib.pyplot as plt

    import astropy.units as u
    from lightkurve import search_lightcurve, LightkurveWarning

    from gadfly import StellarOscillatorKernel, Hyperparameters, PowerSpectrum

    # Some (randomly chosen) real stars from Huber et al. (2011)
    # https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract
    kics = [9333184, 8416311, 8624155, 3120486, 9650527]
    masses = [0.9, 1.5, 1.8, 1.9, 2.0] * u.M_sun
    radii = [10.0, 2.2, 8.8, 6.7, 10.9] * u.R_sun
    temperatures = [4919, 6259, 4944, 4929, 4986] * u.K
    luminosities = [52.3, 6.9, 41.2, 23.9, 65.4] * u.L_sun

    fig, axes = plt.subplots(len(kics), 1, figsize=(10, 14))

    stellar_props = [kics, masses, radii, temperatures, luminosities, axes]

Next, let's write a function for interacting with lightkurve:

.. code-block:: python

    def get_light_curve(target_name)
        """
        Download the light curve for a Kepler target with
        name ``target_name``. Tries to get short cadence
        first, then long cadence second.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LightkurveWarning)
            # first try for short cadence:
            lc = search_lightcurve(
                target_name, mission='Kepler', cadence='short'
            ).download_all()

            # if there is no short cadence, try long:
            if lc is None:
                lc = search_lightcurve(
                    target_name, mission='Kepler', cadence='long'
                ).download_all()
        return lc

Now we'll call a big loop to do most of the work:

.. code-block:: python

    # iterate over each star:
    for i, (kic, mass, radius, temperature, luminosity, axis) in enumerate(zip(*stellar_props)):
        # scale the set of solar hyperparameters for each
        # Kepler star, given their (spectroscopic) stellar parameters
        hp = Hyperparameters.for_star(
            mass, radius, temperature, luminosity, quiet=True
        )

        # Assemble a celerite2-compatible kernel for the star:
        kernel = StellarOscillatorKernel(hp)

        # Get the full Kepler light curve for the target:
        target_name = f'KIC {kic}'

        # download the Kepler light curve:
        lc = get_light_curve(target_name)

        # Compute the power spectrum:
        ps = PowerSpectrum.from_light_curve(
            lc,
            name=target_name,
            detrend_poly_order=1
        )

        # Plot the binned PSD and the kernel PSD. This plot function
        # takes lots of keyword arguments so you can fine-tune your
        # plots:
        obs_kw = dict(color='k', marker='.', lw=0)

        ps.bin(600).plot(
            ax=axis,
            kernel=kernel,
            freq=ps.frequency,
            obs_kwargs=obs_kw,
            legend=True,
            n_samples=5e3,
            label_kernel='Pred. kernel',
            label_obs=target_name,
            kernel_kwargs=dict(color=f'C{i}', alpha=0.9),
            title=""
        )

        # Gray out a region at frequencies < 1 / month, which will show
        # a decrease in power caused by the detrending:
        kepler_cutoff_frequency = (1 / (30 * u.day)).to(u.uHz).value
        axis.axvspan(0, kepler_cutoff_frequency, color='silver', alpha=0.1)
        axis.set_xlim(1e-1, 1e4)
        axis.set_ylim(
            np.nanmin(ps.power.value) / 5,
            np.nanmax(ps.power.value) * 5
        )
    fig.tight_layout()

Ok, let's see the output:

.. plot::

    import warnings
    import numpy as np
    import matplotlib.pyplot as plt

    from gadfly import StellarOscillatorKernel, Hyperparameters, PowerSpectrum

    import astropy.units as u
    from lightkurve import search_lightcurve, LightkurveWarning

    # Some (randomly chosen) real stars from Huber et al. (2011)
    # https://ui.adsabs.harvard.edu/abs/2011ApJ...743..143H/abstract
    kics = [9333184, 8416311, 8624155, 3120486, 9650527]
    masses = [0.9, 1.5, 1.8, 1.9, 2.0] * u.M_sun
    radii = [10.0, 2.2, 8.8, 6.7, 10.9] * u.R_sun
    temperatures = [4919, 6259, 4944, 4929, 4986] * u.K
    luminosities = [52.3, 6.9, 41.2, 23.9, 65.4] * u.L_sun

    fig, axes = plt.subplots(len(kics), 1, figsize=(10, 14))

    stellar_props = [kics, masses, radii, temperatures, luminosities, axes]

    # iterate over each star:
    for i, (kic, mass, radius, temperature, luminosity, axis) in enumerate(zip(*stellar_props)):
        # scale the set of solar hyperparameters for each
        # Kepler star, given their (spectroscopic) stellar parameters
        hp = Hyperparameters.for_star(
            mass, radius, temperature, luminosity, quiet=True
        )

        # Assemble a celerite2-compatible kernel for the star:
        kernel = StellarOscillatorKernel(hp)

        # Get the full Kepler light curve for the target:
        target_name = f'KIC {kic}'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LightkurveWarning)
            # first try for short cadence:
            lc = search_lightcurve(
                target_name, mission='Kepler', cadence='short'
            ).download_all()

            # if there is no short cadence, try long:
            if lc is None:
                lc = search_lightcurve(
                    target_name, mission='Kepler', cadence='long'
                ).download_all()

        # Compute the power spectrum:
        ps = PowerSpectrum.from_light_curve(
            lc, name=target_name,
            detrend_poly_order=1
        )

        obs_kw = dict(color='k', marker='.', lw=0)

        # Plot the binned PSD of the light curve:
        ps.bin(600).plot(
            ax=axis,
            kernel=kernel,
            freq=ps.frequency,
            legend=True,
            n_samples=5e3,
            label_kernel='Pred. kernel',
            label_obs=target_name,
            obs_kwargs=obs_kw,
            kernel_kwargs=dict(color=f'C{i}', alpha=0.9),
            title=""
        )

        # Gray out a region at frequencies > 1 / month, which are
        # attenuated by detrending:
        kepler_cutoff_frequency = (1 / (30 * u.day)).to(u.uHz).value
        axis.axvspan(0, kepler_cutoff_frequency, color='silver', alpha=0.1)
        axis.set_xlim(1e-1, 1e4)
        axis.set_ylim(0.1, 5 * np.nanmax(ps.power.value))

    fig.tight_layout()

The p-modes are shifting in frequency and amplitude, and the separation between
peaks in the p-modes is scaling with stellar parameters, too. The granulation features
also shift in frequency and amplitude. The kernel PSD (in color) and observations (in black)
begin to diverge at low frequencies because detrending applied to the Kepler time
series tends to remove power at frequencies <0.4 microHz
(equivalent to periods >30 days).

Shot noise
**********

In the examples above, the power spectral density in each ``gadfly`` kernel falls off
rapidly with increasing frequency, since ultimately we have modeled the Sun as a
sum of simple harmonic oscillators. Photometric observations taken with a perfect
instrument, observing a non-rotating star, could still be expected to have a Gaussian
white noise component in their power spectra due to Poisson error (shot noise). In
practice, this imposes a "floor" at high frequencies, where shot noise kicks in and
prevents the rapid decline of the observed stellar power spectrum.

We can approximate this behavior in ``gadfly`` with a convenience kernel called
:py:class:`~gadfly.ShotNoiseKernel`.  If you have a Kepler light curve downloaded
from MAST, like the ones you can get from :py:func:`~lightkurve.search_lightcurve`,
you can pass it along to the :py:meth:`~gadfly.ShotNoiseKernel.from_kepler_light_curve`
class method to create a simple floor in your stellar oscillator kernel. We can now adapt
the example above by editing the block within the big loop to create our composite
kernel:

.. code-block:: python

        light_curve = search_lightcurve(
            name, mission='Kepler', quarter=range(6), cadence='short'
        ).download_all()

        kernel = (
            # kernel for scaled stellar oscillations and granulation
            StellarOscillatorKernel(hp) +

            # add in a kernel for Kepler shot noise
            ShotNoiseKernel.from_kepler_light_curve(light_curve)
        )

Let's now run the adapted code on the first two stars (click the "Source code" link below
to see the code that generates this plot):

.. plot::

    import warnings
    import numpy as np
    import matplotlib.pyplot as plt

    import astropy.units as u
    from lightkurve import search_lightcurve

    from gadfly import (
        Hyperparameters, PowerSpectrum,
        StellarOscillatorKernel, ShotNoiseKernel
    )

    # Some (randomly chosen) real stars from Huber et al. (2011)
    kics = [9333184, 8416311]
    masses = [0.9, 1.5] * u.M_sun
    radii = [10.0, 2.2] * u.R_sun
    temperatures = [4919, 6259] * u.K
    luminosities = [52.3, 6.9] * u.L_sun
    cadences = [30, 1] * u.min

    fig, axes = plt.subplots(len(kics), 1, figsize=(8, 5), sharex=True)

    stellar_props = [
        kics, masses, radii, temperatures,
        luminosities, cadences, axes
    ]

    # iterate over each star:
    for i, (kic, mass, rad, temp, lum, cad, axis) in enumerate(zip(*stellar_props)):
        name = f'KIC {kic}'
        hp = Hyperparameters.for_star(
            mass, rad, temp, lum, name=name, quiet=True
        )

        cadence_str = 'short' if cad == 1*u.min else 'long'
        light_curve = search_lightcurve(
            name, mission='Kepler', quarter=range(6),
            cadence=cadence_str
        ).download_all()

        kernel = (
            # kernel for scaled stellar oscillations and granulation
            StellarOscillatorKernel(hp) +

            # add in a kernel for Kepler shot noise
            ShotNoiseKernel.from_kepler_light_curve(light_curve)
        )

        ps = PowerSpectrum.from_light_curve(
            light_curve, name=name,
            detrend_poly_order=1
        ).bin(200)

        kernel_kw = dict(color=f"C{i}", alpha=0.9)
        obs_kw = dict(color='k', marker='.', lw=0)
        freq = np.logspace(-0.5, 4, int(1e3)) * u.uHz
        kernel.plot(
            ax=axis,
            p_mode_inset=False,
            freq=freq,
            obs=ps,
            obs_kwargs=obs_kw,
            kernel_kwargs=kernel_kw,
            n_samples=1e4,
            title=""
        )

        lower_y = np.nanmin(ps.power).value / 3
        lower_x = (1 / (30 * u.d)).to(u.uHz).value
        axis.set_ylim([lower_y, None])
        axis.set_xlim([lower_x, None])
        if i < len(kics) - 1:
            axis.set_xlabel(None)
    fig.tight_layout()
