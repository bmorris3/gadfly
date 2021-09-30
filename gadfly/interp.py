import numpy as np

__all__ = ['interpolate_missing_data']


def interpolate_missing_data(times, fluxes, cadences=None):
    """
    Assuming ``times`` are uniformly spaced with missing cadences,
    fill in the missing cadences with linear interpolation.

    Cadences can be passed if they are known.

    Parameters
    ----------
    times : numpy.ndarray
        Incomplete but otherwise uniformly sampled times
    fluxes : numpy.ndarray
        Flux for each time in ``times``
    cadences : numpy.ndarray, optional
        Integer cadence number of each observation.

    Returns
    -------
    interpolated_times : numpy.ndarray
        ``times`` with filled-in missing cadences
    interpolated_fluxes : numpy.ndarray
        ``fluxes`` with filled-in missing cadences
    """
    first_time = times[0]

    if cadences is not None:
        # Median time between cadences
        dt = np.median(np.diff(times) / np.diff(cadences))
        cadence_indices = cadences - cadences[0]
    else:
        # Find typical time between cadences:
        dt = np.median(np.diff(times))
        # Approximate the patchy grid of integer cadence indices,
        # i.e.: (0, 1, 3, 4, 5, 8, ...)
        cadence_indices = np.rint((times - first_time)/dt)

    # Find missing cadence indices if that grid were complete
    expected_cadence_indices = set(np.arange(cadence_indices.min(),
                                             cadence_indices.max()))
    missing_cadence_indices = expected_cadence_indices.difference(set(cadence_indices))
    # Convert the missing cadences to times
    missing_times = first_time + np.array(list(missing_cadence_indices))*dt

    # Interpolate to find fluxes at missing times
    interp_fluxes = np.interp(missing_times, times, fluxes)

    # Combine the interpolated and input times, fluxes
    interpolated_fluxes = np.concatenate([fluxes, interp_fluxes])
    interpolated_times = np.concatenate([times, missing_times])

    # Sort the times, fluxes, so that you can compute the ACF on them:
    sort_by_time = np.argsort(interpolated_times)
    interpolated_fluxes = interpolated_fluxes[sort_by_time]
    interpolated_times = interpolated_times[sort_by_time]
    return interpolated_times, interpolated_fluxes