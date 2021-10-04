====================================
Stellar oscillations and granulation
====================================

Let's synthesize a time series of 60-second cadence observations of the
Sun-like star spanning 100 days of observations:

.. code-block:: python

    from gadfly import generate_stellar_fluxes
    import astropy.units as u

    time, flux, kernel = generate_stellar_fluxes(
        duration=100 * u.day,
        M=1.01 * u.M_sun,
        T_eff=5900 * u.K,
        R=1.01 * u.R_sun,
        L=1.05 * u.L_sun,
        cadence=60 * u.s
    )

    plt.plot(time.to(u.day).value, 1e6 * flux, '.k')
    plt.gca().set(
        xlabel='Time [d]', ylabel='Flux [ppm]'
    )
    plt.show()


.. plot::

    from gadfly import generate_stellar_fluxes
    import astropy.units as u

    time, flux, kernel = generate_stellar_fluxes(
        duration=100 * u.day,
        M=1.01 * u.M_sun,
        T_eff=5900 * u.K,
        R=1.01 * u.R_sun,
        L=1.05 * u.L_sun,
        cadence=60 * u.s
    )

    plt.plot(time.to(u.day).value, 1e6 * flux, '.k')
    plt.gca().set(
        xlabel='Time [d]', ylabel='Flux [ppm]'
    )
    plt.show()


and let's plot the power spectrum of those simulated observations:

.. code-block:: python

    from gadfly.psd import power_spectrum
    import matplotlib.pyplot as plt

    freq, power = power_spectrum(flux)

    plt.loglog(
        1e6 * freq,
        kernel.get_psd(2 * np.pi * freq) * (1e6 / 2 / np.pi),
        'r', label='Kernel'
    )
    plt.loglog(1e6 * freq, power, ',k', label='Sim. obs.')
    plt.ylim([1e-7, 1e3])
    plt.legend()
    plt.gca().set(
        xlabel='Freq [$\mu$Hz]', ylabel='Power [ppm$^2$ Hz$^{-1}$]'
    )
    plt.show()

.. plot::

    from gadfly import generate_stellar_fluxes
    import astropy.units as u

    time, flux, kernel = generate_stellar_fluxes(
        duration=100 * u.day,
        M=1.01 * u.M_sun,
        T_eff=5900 * u.K,
        R=1.01 * u.R_sun,
        L=1.05 * u.L_sun,
        cadence=60 * u.s
    )

    from gadfly.psd import power_spectrum
    import matplotlib.pyplot as plt

    freq, power = power_spectrum(flux)

    plt.loglog(
        1e6 * freq,
        kernel.get_psd(2 * np.pi * freq) * (1e6 / 2 / np.pi),
        'r', label='Kernel'
    )
    plt.loglog(1e6 * freq, power, ',k', label='Sim. obs.')
    plt.ylim([1e-7, 1e3])
    plt.legend()
    plt.gca().set(
        xlabel='Freq [$\mu$Hz]', ylabel='Power [ppm$^2$ Hz$^{-1}$]'
    )
    plt.show()
