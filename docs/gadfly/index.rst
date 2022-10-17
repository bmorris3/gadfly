###############
Getting started
###############

.. plot::
    :include-source:

    from gadfly import SolarOscillatorKernel, PowerSpectrum
    from gadfly.sun import download_soho_virgo_time_series
    import astropy.units as u

    # Download a year of the total solar irradiance observations
    # from SOHO VIRGO PMO6:
    times, fluxes, delta_t = download_soho_virgo_time_series(
        full_time_series=False
    )

    # Compute the power spectrum from the SOHO time series
    ps = PowerSpectrum.from_light_curve(fluxes, delta_t)

    # Generate a celerite2 kernel that approximates the solar
    # power spectrum
    kernel = SolarOscillatorKernel()

    # Plot the kernel PSD, and the observed solar PSD:
    fig, ax = kernel.plot(
        observed_power_spectrum=ps.bin(100),
        obs_kwargs=dict(marker='o', lw=0)
    )