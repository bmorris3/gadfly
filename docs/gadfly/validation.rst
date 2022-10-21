Validation with Kepler
======================

Outline
-------

All of those scaling relations sound great, but do they
yield kernels that accurately reproduce Kepler time series
photometric observations of stars? Let's find out!

In :ref:`getting_started`, we wrote out the
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
3. Compute a :py:class:`~gadfly.PowerSpectrum` from the light curve
   with the :py:meth:`~gadfly.PowerSpectrum.from_light_curve` method.
4. Plot the results for each star.

Here we go!

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

    fig, axes = plt.subplots(len(kics), 1, figsize=(10, 12))

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
            mass, radius, temperature, luminosity
        )

        # Assemble a celerite2-compatible kernel for the star:
        kernel = StellarOscillatorKernel(hp)

        # Get the full Kepler light curve for the target:
        target_name = f'KIC {kic}'

        # download the Kepler light curve:
        lc = get_light_curve(target_name)

        # Compute the power spectrum, bin it up into 600 bins:
        ps = PowerSpectrum.from_light_curve(
            lc, interpolate_and_detrend=True,
            name=target_name,
            detrend_poly_order=3
        ).bin(600)

        # Plot the binned PSD and the kernel PSD. This plot function
        # takes lots of keyword arguments so you can fine-tune your
        # plots:
        ps.plot(
            ax=axis,
            kernel=kernel,
            freq=ps.frequency,
            legend=True,
            p_mode_inset=False,
            n_samples=5e3,
            label_kernel='kernel',
            label_obs=target_name,
            obs_kwargs=dict(marker='.', markersize=3, color='k', lw=0),
            kernel_kwargs=dict(color=f'C{i}', alpha=0.7),
            title=""
        )

        # Gray out a region at frequencies < 1 / month, which show a
        # decrease in power caused by the detrending:
        kepler_quarter_frequency = (1 / (30 * u.day)).to(u.uHz).value
        axis.axvspan(0, kepler_quarter_frequency, color='silver', alpha=0.1)
        axis.set_xlim(([1e-1, 1e4]*u.uHz).value)
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

    fig, axes = plt.subplots(len(kics), 1, figsize=(10, 12))

    stellar_props = [kics, masses, radii, temperatures, luminosities, axes]

    # iterate over each star:
    for i, (kic, mass, radius, temperature, luminosity, axis) in enumerate(zip(*stellar_props)):
        # scale the set of solar hyperparameters for each
        # Kepler star, given their (spectroscopic) stellar parameters
        hp = Hyperparameters.for_star(
            mass, radius, temperature, luminosity
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

        # Compute the power spectrum, bin it:
        ps = PowerSpectrum.from_light_curve(
            lc, interpolate_and_detrend=True, name=target_name,
            detrend_poly_order=3
        ).bin(600)

        # Plot the binned PSD of the light curve:
        ps.plot(
            ax=axis,
            kernel=kernel,
            freq=ps.frequency,
            legend=True,
            p_mode_inset=False,
            n_samples=5e3,
            label_kernel='kernel',
            label_obs=target_name,
            obs_kwargs=dict(marker='.', markersize=3, color='k', lw=0),
            kernel_kwargs=dict(color=f'C{i}', alpha=0.7),
            title=""
        )

        # Gray out a region at frequencies < 1 / month, which are
        # attenuated by detrending:
        kepler_quarter_frequency = (1 / (30 * u.day)).to(u.uHz).value
        axis.axvspan(0, kepler_quarter_frequency, color='silver', alpha=0.1)
        axis.set_xlim(([1e-1, 1e4]*u.uHz).value)
        axis.set_ylim(
            np.nanmin(ps.power.value) / 5,
            np.nanmax(ps.power.value) * 5
        )
    fig.tight_layout()

The p-modes are shifting in frequency and amplitude, and the separation between
peaks in the p-modes is scaling with stellar parameters, too. The granulation features
also shift in frequency and amplitude. The kernel PSD (in color) and observations (in black)
begin to diverge at low frequencies because detrending applied to the Kepler time
series tends to remove power at lower frequencies than ~0.4:math:`\mu`Hz
(equivalent to a period of 30 days).
