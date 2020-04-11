import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astroquery.jplhorizons import Horizons


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

    eph = obj.ephemerides(quantities="1,13", extra_precision=True)
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
            np.interp(times.mjd, times_jpl.mjd, partial), dtype=bool,
        )
        data["ecl_tot"] = np.array(
            np.interp(times.mjd, times_jpl.mjd, umbra), dtype=bool,
        )
        data["occ_umbra"] = np.array(
            np.interp(times.mjd, times_jpl.mjd, occ_umbra), dtype=bool,
        )
        data["occ_sun"] = np.array(
            np.interp(times.mjd, times_jpl.mjd, occ_sunlight), dtype=bool,
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
            times.mjd, times_jpl.mjd, eph["NPole_ang"].to(u.rad),
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
