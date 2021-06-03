import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astroquery.jplhorizons import Horizons
from scipy.interpolate import interp1d

import starry

starry.config.lazy = False


def get_body_ephemeris(
    times, body_id="501", step="1m", return_orientation=True
):
    """
    Given a NAIF code of a Solar System body, this function computes the
    position and orientation of the body in equatorial coordinates at requested
    times using ephemeris from JPL Horizons. The relevant JPL Horizons data is
    the following:

    'RA', 'DEC'
    ###########
    Position on the plane of the sky.

    'ang_width'
    ###########
    The angle subtended by the disk of the target seen by the observer, if
    it was fully illuminated. The target diameter is taken to be the IAU2009
    equatorial diameter. Oblateness aspect is not currently included.
    Units: ARCSECONDS

    'sat_vis'
    #########
    The angle between the center of a non-lunar target body and the center
    of the primary body it revolves around, as seen by the observer.
    Units: ARCSECONDS

    Non-lunar natural satellite visibility codes (limb-to-limb):

    /t = Transitting primary body disk, /O = Occulted by primary body disk,
    /p = Partial umbral eclipse,        /P = Occulted partial umbral eclipse,
    /u = Total umbral eclipse,          /U = Occulted total umbral eclipse,
    /- = Target is the primary body,    /* = None of above ("free and clear")

    ... the radius of major bodies is taken to be the equatorial value (max)
    defined by the IAU2009 system. Atmospheric effects and oblateness aspect
    are not currently considered in these computations. Light-time is included.

    'r'
    ###
    Heliocentric range ("r", light-time compensated) and range-rate ("rdot")
    of the target point at the instant light later seen by the observer at
    print-time would have left the target (at the instant print-time minus
    down-leg light-time); the Sun-to-target distance traveled by a ray of
    light emanating from the center of the Sun that reaches the target at some
    instant and is recordable by the observer one down-leg light-time later at
    print-time. "rdot" is a projection of the velocity vector along this ray,
    the light-time-corrected line-of-sight from the Sun's center, and indicates
    relative motion. A positive "rdot" means the target is moving away from
    the Sun. A negative "rdot" means the target is moving toward the Sun.
    Units: AU or KM, KM/S

    'PDObsLong', 'PDObsLat'
    #######################
    Apparent planetodetic ("geodetic") longitude and latitude (IAU2009
    model) of the center of the target seen by the OBSERVER at print-time.
    This is NOT exactly the same as the "sub-observer" (nearest) point for
    a non-spherical target shape, but is generally very close if not a highly
    irregular body shape. Light travel-time from target to observer is taken
    into account. Latitude is the angle between the equatorial plane and the
    line perpendicular to the reference ellipsoid of the body. The reference
    ellipsoid is an oblate spheroid with a single flatness coefficient in
    which the y-axis body radius is taken to be the same value as the x-axis
    radius. For the gas giants only (Jupiter, Saturn, Uranus and Neptune),
    these longitudes are based on the Set III prime meridian angle, referred
    to the planet's rotating magnetic field. Latitude is always referred to
    the body dynamical equator.  Note there can be an offset between the
    dynamical pole and the magnetic pole. Positive longitude is to the WEST.
    Units: DEGREES

    'PDSunLong', 'PDSunLat'
    #######################
    Solar sub-long & sub-lat
    Apparent planetodetic ("geodetic") longitude and latitude of the Sun
    (IAU2009) as seen by the observer at print-time.  This is NOT exactly the
    same as the "sub-solar" (nearest) point for a non-spherical target shape,
    but is generally very close if not a highly irregular body shape. Light
    travel-time from Sun to target and from target to observer is taken into
    account.  Latitude is the angle between the equatorial plane and the line
    perpendicular to the reference ellipsoid of the body. The reference
    ellipsoid is an oblate spheroid with a single flatness coefficient in
    which the y-axis body radius is taken to be the same value as the x-axis
    radius. For the gas giants only (Jupiter, Saturn, Uranus and Neptune),
    these longitudes are based on the Set III prime meridian angle, referred
    to the planet's rotating magnetic field. Latitude is always referred to
    the body dynamical equator. Note there can be an offset between the
    dynamical pole and the magnetic pole. Positive longitude is to the WEST.
    Units: DEGREES

    'NPole_ang'
    #######################
    Target's North Pole position angle (CCW with respect to direction of
    true-of-date Celestial North Pole) and angular distance from the
    "sub-observer" point (center of disk) at print time. Negative distance
    indicates N.P. on hidden hemisphere. Units: DEGREES and ARCSECONDS

    Args:
    times (astropy.time): Observation times.
    body_id (str, optional): NAIF code for the target body. By default '501'
        for Io.
    step (str, optional): Step size for querying JPL Horizons. Minimum is "1m".
        Make sure this is sufficiently small for accurate ephemeris.
    return_orientation (bool, optional): If False, the function returns only
        the position and the angular diameter of the target body, otherwise the
        function returns all parameters needed to determine the location of a
        Starry map in reflected light. By default True.

    Returns:
        astropy.timeseries.TimeSeries

        An astropy.TimeSeries object with entries relevant for defining the
        orientation of a Starry map at any given time. All quantities are
        interpolated from the JPL Horizons data to the requested times. The
        columns are the following: RA [arcsec], DEC [arcsec], inc [deg],
        obl [deg], theta [deg], xs [AU], ys[AU], zs[AU], par_ecl[bool],
        tot_ecl[bool], occ_primary[bool] where the (xs, ys, zs) are the
        Cartesian coordinates
        of the Sun w.r. to target body as seen from Earth. Boolean flag
        'par_ecl' is True
        whenever the target body is in a partial eclipse with respect to its
        primary and similarly for a total eclipse. The flags 'ecl_tot' and
        'ecl_par' denote the total and partial eclipses with respect to the
        primary body of the target body. The flags 'occ_umbra' and 'occ_sun'
        denote times when the target body is occulted by the primary either
        while eclipsed or in sunlight. For example, the flag 'occ_umbra' will
        capture all times when the target body is either entering an occultation
        while eclipsed or exiting an occultation into an eclipse. These flag
        don't capture mutual occultations between the satellites.
    """

    start = times.isot[0]

    # because Horizons time range doesn't include the endpoint we need to add
    # some extra time
    if step[-1] == "m":
        padding = 2 * float(step[:-1]) / (60 * 24)
    elif step[-1] == "h":
        padding = 2 * float(step[:-1]) / 24
    elif step[-1] == "d":
        padding = 2 * float(step[:-1])
    else:
        raise ValueError(
            "Unrecognized JPL Horizons step size. Use '1m' or '1h' for example."
        )

    end = Time(times.mjd[-1] + padding, format="mjd").isot

    # Query JPL Horizons
    epochs = {"start": start, "stop": end, "step": step}
    obj = Horizons(id=body_id, epochs=epochs, id_type="id")

    eph = obj.ephemerides(extra_precision=True)
    times_jpl = Time(eph["datetime_jd"], format="jd")

    # Store all data in a TimeSeries object
    data = TimeSeries(time=times)

    data["RA"] = (
        np.interp(times.mjd, times_jpl.mjd, eph["RA"]) * eph["RA"].unit
    )
    data["DEC"] = (
        np.interp(times.mjd, times_jpl.mjd, eph["DEC"]) * eph["DEC"].unit
    )
    data["ang_width"] = (
        np.interp(times.mjd, times_jpl.mjd, eph["ang_width"])
        * eph["ang_width"].unit
    )

    data["dist"] = (
        np.interp(times.mjd, times_jpl.mjd, eph["delta"]) * eph["delta"].unit
    )

    # If the target body is Jupiter, compute the equatorial angular diameter
    if body_id == "599":
        dist = eph["delta"].to(u.km)
        jup_eq_rad = 71492 * u.km
        ang_width = 2 * jup_eq_rad / dist
        data["ang_width"] = (
            np.interp(times.mjd, times_jpl.mjd, np.array(ang_width)) * u.rad
        ).to(u.arcsec)

    if return_orientation:
        eph = obj.ephemerides(extra_precision=True)

        # Boolean flags for occultations/eclipses
        occ_sunlight = eph["sat_vis"] == "O"
        umbra = eph["sat_vis"] == "u"
        occ_umbra = eph["sat_vis"] == "U"

        partial = eph["sat_vis"] == "p"
        occ_partial = eph["sat_vis"] == "P"

        occulted = np.any([occ_umbra, occ_sunlight], axis=0)

        data["ecl_par"] = np.array(
            np.interp(times.mjd, times_jpl.mjd, partial),
            dtype=bool,
        )
        data["ecl_tot"] = np.array(
            np.interp(times.mjd, times_jpl.mjd, umbra),
            dtype=bool,
        )
        data["occ_umbra"] = np.array(
            np.interp(times.mjd, times_jpl.mjd, occ_umbra),
            dtype=bool,
        )
        data["occ_sun"] = np.array(
            np.interp(times.mjd, times_jpl.mjd, occ_sunlight),
            dtype=bool,
        )

        # Helper functions for dealing with angles and discontinuities
        subtract_angles = (
            lambda x, y: np.fmod((x - y) + np.pi * 3, 2 * np.pi) - np.pi
        )

        def interpolate_angle(x, xp, yp):
            """
            Interpolate an angular quantity on domain [-pi, pi) and avoid
            discountinuities.
            """
            cosy = np.interp(x, xp, np.cos(yp))
            siny = np.interp(x, xp, np.sin(yp))

            return np.arctan2(siny, cosy)

        # Inclination of the starry map = 90 - latitude of the central point of
        # the observed disc
        data["inc"] = interpolate_angle(
            times.mjd,
            times_jpl.mjd,
            np.pi / 2 * u.rad - eph["PDObsLat"].to(u.rad),
        ).to(u.deg)

        # Rotational phase of the starry map is the observer longitude
        data["theta"] = (
            interpolate_angle(
                times.mjd,
                times_jpl.mjd,
                eph["PDObsLon"].to(u.rad) - np.pi * u.rad,
            ).to(u.deg)
        ) + 180 * u.deg

        # Obliquity of the starry map is the CCW angle from the celestial
        # NP to the NP of the target body
        data["obl"] = interpolate_angle(
            times.mjd,
            times_jpl.mjd,
            eph["NPole_ang"].to(u.rad),
        ).to(u.deg)

        # Compute the location of the subsolar point relative to the central
        # point of the disc
        lon_subsolar = subtract_angles(
            np.array(eph["PDSunLon"].to(u.rad)),
            np.array(eph["PDObsLon"].to(u.rad)),
        )
        lon_subsolar = 2 * np.pi - lon_subsolar  # positive lon. is to the east

        lat_subsolar = subtract_angles(
            np.array(eph["PDSunLat"].to(u.rad)),
            np.array(eph["PDObsLat"].to(u.rad)),
        )

        # Location of the subsolar point in cartesian Starry coordinates
        xs = np.array(eph["r"]) * np.cos(lat_subsolar) * np.sin(lon_subsolar)
        ys = np.array(eph["r"]) * np.sin(lat_subsolar)
        zs = np.array(eph["r"]) * np.cos(lat_subsolar) * np.cos(lon_subsolar)

        data["xs"] = np.interp(times.mjd, times_jpl.mjd, xs) * u.AU
        data["ys"] = np.interp(times.mjd, times_jpl.mjd, ys) * u.AU
        data["zs"] = np.interp(times.mjd, times_jpl.mjd, zs) * u.AU

    return data


def get_body_vectors(times, body_id="501", step="1m", location="@sun"):
    """
    Returns the JPL Horizons position (and velocity) vector of a given Solar
    System body for the requested times.

    Args:
        times (astropy.time): Observation times.
    body_id (str, optional): NAIF code for the target body. By default '501'
        for Io.
    step (str, optional): Step size for querying JPL Horizons. Minimum is "1m".
        Make sure this is sufficiently small for accurate ephemeris.
    location (str, optional): The origin of the coordinate system. By default
        "@sun" for heliocentric position vectors. Other options include "500"
        for center of earth and "@ssb" for Solar System Barycenter.

    Returns:
        astropy.timeseries.TimeSeries

        An astropy.TimeSeries object specifying the (x, y, z) coordinates and
        (vx, vy, vz) velocity components and distance r of the target body.
    """

    start = times.isot[0]

    # because Horizons time range doesn't include the endpoint we need to add
    # some extra time
    if step[-1] == "m":
        padding = 2 * float(step[:-1]) / (60 * 24)
    elif step[-1] == "h":
        padding = 2 * float(step[:-1]) / 24
    elif step[-1] == "d":
        padding = 2 * float(step[:-1])
    else:
        raise ValueError(
            "Unrecognized JPL Horizons step size. Use '1m' or '1h' for example."
        )

    end = Time(times.mjd[-1] + padding, format="mjd").isot

    # Query JPL Horizons
    epochs = {"start": start, "stop": end, "step": step}
    obj = Horizons(id=body_id, epochs=epochs, id_type="id", location=location)

    vec = obj.vectors()
    times_jpl = Time(vec["datetime_jd"], format="jd")

    # Store all data in a TimeSeries object
    data = TimeSeries(time=times)

    data["x"] = np.interp(times.mjd, times_jpl.mjd, vec["x"]) * vec["x"].unit
    data["y"] = np.interp(times.mjd, times_jpl.mjd, vec["y"]) * vec["y"].unit
    data["z"] = np.interp(times.mjd, times_jpl.mjd, vec["z"]) * vec["z"].unit
    data["vx"] = (
        np.interp(times.mjd, times_jpl.mjd, vec["vx"]) * vec["vx"].unit
    )
    data["vy"] = (
        np.interp(times.mjd, times_jpl.mjd, vec["vy"]) * vec["vy"].unit
    )
    data["vz"] = (
        np.interp(times.mjd, times_jpl.mjd, vec["vz"]) * vec["vz"].unit
    )
    data["r"] = (
        np.interp(times.mjd, times_jpl.mjd, vec["range"]) * vec["range"].unit
    )

    return data


def get_effective_jupiter_radius(latitude, method="howell"):
    """
    Return radius of jupiter (in km) at the 2.2 mbar level,
    given the Jupiter planetocentric latitude in degrees.
    """

    if method == "howell":
        # Conversion factors
        dpr = 360.0 / (2.0 * np.pi)
        # Compute powers of sin(lat)
        u0 = 1.0
        u1 = np.sin(latitude / dpr)
        u2 = u1 * u1
        u3 = u1 * u2
        u4 = u1 * u3
        u5 = u1 * u4
        u6 = u1 * u5

        # Legendre polynomials
        p0 = 1
        p1 = u1
        p2 = (-1.0 + 3.0 * u2) / 2.0
        p3 = (-3.0 * u1 + 5.0 * u3) / 2.0
        p4 = (3.0 - 30.0 * u2 + 35.0 * u4) / 8.0
        p5 = (15.0 * u1 - 70.0 * u3 + 63.0 * u5) / 8.0
        p6 = (-5.0 + 105.0 * u2 - 315.0 * u4 + 231.0 * u6) / 16.0

        # Find the radius at 100 mbar
        jr = (
            71541.0
            - 1631.3 * p0
            + 16.8 * p1
            - 3136.2 * p2
            - 6.9 * p3
            + 133.0 * p4
            - 18.9 * p5
            - 8.5 * p6
        )

        jr += 85.0  #  Find the radius at 2.2 mbar
        #  According to J. Spencer, this is the
        #  half light point for occultations.

        jr += -27  # subtract one scale height due to bending of light
        return jr

    else:
        # Radius in km, by eye estimate
        reff_100 = np.array(
            [66896, 67350, 71400, 71541, 70950, 70400, 67950, 66896]
        )
        lat_100 = np.array([-90, -70.0, -10.0, 0.0, 20.0, 27.0, 60.0, 90])

        # Cubic interpolate
        f = interp1d(lat_100, reff_100, kind="cubic", fill_value="extrapolate")
        jr = float(f(latitude))

        # Correction factor for going from 100->2.2mbar
        jr += -np.log(2.2 / 100) * 27 - 27  # scale height 27km
        return jr


def get_occultation_latitude(x_io, y_io, re):
    """
    Compute the approximate latitude of Jupiter at which an occultation occurs.
    In this case we assume that the Jupiter is a sphere with the radius equal
    to the equatorial radius. This assumption is good enough for computing
    the latitude because occultations occur at around +-20 or so degrees N.
    """
    y = x_io
    z = y_io
    x = np.sqrt(re ** 2 - z ** 2 - y ** 2)
    ang = np.sqrt(x ** 2 + y ** 2) / z
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    return 90 - theta * 180 / np.pi


def get_occultor_position_and_radius(
    eph_occulted,
    eph_occultor,
    occultor_is_jupiter=False,
    rotate=True,
    return_occ_lat=False,
    **kwargs
):
    """
    Given the ephemeris of an occulted object and an occultor, the function
    returns the relative position of the occultor in Starry format. If the
    occultor is Jupiter the radius isn't trivial to compute because Jupiter is
    an oblate spheroid. In that case we instead compute the effective radius at
    a given planetocentric latitude at which the occultation is happening.
    This is implemented in :func:`get_jupiter_effective_radius`.

    Args:
        eph_occulted (astropy.timeseries.TimeSeries): ephemeris of the occulted
            body.
        eph_occultor (astropy.timeseries.TimeSeries): ephemeris of the occultor.
        occultor_is_jupiter (bool): Set to true if occultor is Jupiter because
            Jupiter is non-spherical and its radius needs to be estimated in a
            different way. Defaults to False.
        rotate (bool): Rotate the position vectors to a frame in which the
            obliquity of the occultor is zero. Defaults to True.
        return_occ_lat (bool): Optionally return the occultation latitude if
            the occultor is Jupiter. Defaults to False.

    Returns:
        list: (xo, yo, ro)
    """
    delta_ra = (eph_occultor["RA"] - eph_occulted["RA"]).to(u.arcsec)
    delta_dec = (eph_occultor["DEC"] - eph_occulted["DEC"]).to(u.arcsec)

    xo = (
        -delta_ra
        * np.cos(eph_occulted["DEC"].to(u.rad))
        / (0.5 * eph_occulted["ang_width"].to(u.arcsec))
    ).value
    yo = delta_dec / (0.5 * eph_occulted["ang_width"].to(u.arcsec)).value

    if occultor_is_jupiter is False:
        # Convert everything to units where the radius of Io = 1
        rad_occ = eph_occultor["ang_width"].to(u.arcsec) / eph_occulted[
            "ang_width"
        ].to(u.arcsec)
        ro = np.mean(rad_occ).value

    # Jupiter is non-spherical so we need to compute an effective radius
    else:
        re = 71541 + 59  # equatorial radius of Jupiter (approx) [km]
        r_io = 1821.3 * u.km  # radius of Io (km)
        jup_dist = eph_occultor["dist"].to(u.km)

        xo = (
            ((-delta_ra * np.cos(eph_occultor["DEC"].to(u.rad)))).to(u.rad)
            * jup_dist
            / r_io
        )
        yo = delta_dec.to(u.rad) * jup_dist / r_io

        obl = eph_occulted["obl"]
        inc = np.mean(eph_occultor["inc"])

        xo_unrot = xo.value
        yo_unrot = yo.value

        # Rotate to coordinate system where the obliquity of Io is 0
        theta_rot = -obl.to(u.rad).value
        xo_rot, yo_rot = rotate_vectors(
            np.array(xo_unrot), np.array(yo_unrot), np.array(theta_rot)
        )

        # Choose point inside Jupiter
        idx = np.argmin(np.abs(xo_rot))

        # Position of Io relative to Jupiter in km
        x_io = -xo_rot[idx] * r_io.value
        y_io = -yo_rot[idx] * r_io.value

        lat = get_occultation_latitude(x_io, y_io, re)
        reff = get_effective_jupiter_radius(lat, **kwargs)
        ro = reff / r_io.value

    # Rotate position vefctors such that obliquity of the occultor is 0
    if rotate:
        theta_rot = -eph_occulted["obl"].to(u.rad).value
        xo, yo = rotate_vectors(
            np.array(xo), np.array(yo), np.array(theta_rot)
        )
    else:
        xo = xo
        yo = yo

    if return_occ_lat:
        if not occultor_is_jupiter:
            raise ValueError(
                "Occultation latitude is only defined if",
                " the occultor is Jupiter.",
            )
        return xo, yo, ro, lat
    else:
        return xo, yo, ro


def rotate_vectors(x, y, theta_rot):
    """
    Given a pair of vectors (x, y) representing coordinates on a 2D plane,
    the function returns a pair of vectors of the same length, but rotated by
    an angle `theta_rot` in the counterclockwise direction.

    Args:
        x (numpy.ndarray): x-coordinates of the points.
        y (numpy.ndarray): y-coordinates of the points.
        theta_rot (float): angle in radians.

    Returns:
        list: (x_rot, y_rot)
    """
    c, s = np.cos(theta_rot), np.sin(theta_rot)
    R = np.array(([c, -s], [s, c]))
    r_stacked = np.stack([x, y])

    # Matrix vector multiplication repetead along an axis
    res = np.einsum("ijk,jk->ki", R, r_stacked)

    return res[:, 0], res[:, 1]


def get_smoothing_filter(ydeg, sigma=0.1):
    """
    Returns a smoothing matrix which applies an isotropic Gaussian beam filter
    to a spherical harmonic coefficient vector. This helps suppress ringing
    artefacts around spot like features. The standard deviation of the Gaussian
    filter controls the strength of the smoothing. Features on angular scales
    smaller than ~ 1/sigma are strongly suppressed.

    Args:
        ydeg (int): Degree of the map.
        sigma (float, optional): Standard deviation of the Gaussian filter.
            Defaults to 0.1.

    Returns:
        ndarray: Diagonal matrix of shape (ncoeff, ncoeff) where ncoeff = (l + 1)^2.
    """
    l = np.concatenate([np.repeat(l, 2 * l + 1) for l in range(ydeg + 1)])
    s = np.exp(-0.5 * l * (l + 1) * sigma ** 2)
    S = np.diag(s)
    return S


def get_median_map(
    ydeg,
    samples_ylm,
    projection="Mollweide",
    inc=90,
    theta=0.0,
    nsamples=15,
    resol=300,
    return_std=False,
):
    """
    Given a set of samples from a posterior distribution over the spherical
    harmonic coefficients, the function computes a median map in pixel space.

    Args:
        ydeg (int): Degree of the map.
        samples_ylm (list): List of (amplitude weighted) Ylm samples.
        projection (str, optional): Map projection. Defaults to "Mollweide".
        inc (int, optional): Map inclination. Defaults to 90.
        theta (float, optional): Map phase. Defaults to 0.0.
        nsamples (int, optional): Number of samples to use to compute the
            median. Defaults to 15.
        resol (int, optional): Map resolution. Defaults to 300.
        return_std(bool, optional): If true, the function returns both the
        median map and the standard deviation as a tuple. By default False.

    Returns:
        ndarray: Pixelated map in the requested projection. Shape (resol, resol).
    """
    if len(samples_ylm) < nsamples:
        raise ValueError(
            "Length of Ylm samples list has to be greater than", "nsamples"
        )
    imgs = []
    map = starry.Map(ydeg=ydeg)
    map.inc = inc

    for n in np.random.randint(0, len(samples_ylm), nsamples):
        x = samples_ylm[n]
        map.amp = x[0]
        map[1:, :] = x[1:] / map.amp

        if projection == "Mollweide" or projection == "rect":
            im = map.render(projection=projection, res=resol)
        else:
            im = map.render(theta=theta, res=resol)
        imgs.append(im)

    if return_std:
        return np.nanmedian(imgs, axis=0), np.nanstd(imgs, axis=0)
    else:
        return np.nanmedian(imgs, axis=0)