Getting started
===============

gadfly provides custom Gaussian process kernels
that are useful for approximating solar and stellar irradiance
power spectra and their time series. The kernels are
formulated within the  `celerite2 <https://celerite2.readthedocs.io/>`_
framework.

Solar power spectrum
--------------------

Asteroseismic scaling relations are anchored by the total
solar (TSI) irradiance power spectrum. gadfly's kernels and
their hyperparameters are built on a fit to the `SOHO VIRGO/PMO6
<https://www.pmodwrc.ch/en/research-development/solar-physics/virgo-data-products-archived_webpage/>`_
TSI observations spanning 1996-2016. They were taken at
one-minute cadence and are available online.

Below we download the SOHO VIRGO observations over one year (6 MB),
compute and bin the solar power spectrum, construct a kernel that
approximates the solar power spectrum, and plot both the modeled and
observed power spectra.

.. plot::
    :include-source:

    from gadfly import SolarOscillatorKernel, PowerSpectrum
    from gadfly.sun import download_soho_virgo_time_series

    # Download a year of the total solar irradiance observations
    # from SOHO VIRGO PMO6:
    light_curve = download_soho_virgo_time_series(
        full_time_series=False
    )

    # Compute the power spectrum from the SOHO time series
    ps = PowerSpectrum.from_light_curve(light_curve)

    # Generate a celerite2 kernel that approximates the solar
    # power spectrum
    kernel = SolarOscillatorKernel()

    # Plot the kernel's PSD, and the observed (binned) solar PSD:
    fig, ax = kernel.plot(
        # also plot the observed power spectrum
        obs=ps.bin(bins=100), obs_kwargs=dict(marker='o', lw=0)
    )
