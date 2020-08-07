import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from volcano import utils


def test_get_body_ephemeris():
    # Some random dates
    dates = ["1984-06-01", "2002-03-01", "2015-08-01"]

    for date in dates:
        # Get data from JPL
        start = Time(date, format="isot")
        stop = Time(start.mjd + 1 / 24, format="mjd")

        epochs = {"start": start.isot, "stop": stop.isot, "step": "1m"}

        io = Horizons(id="501", epochs=epochs, id_type="id")
        eph_jpl = io.ephemerides(extra_precision=True)

        times = Time(np.linspace(start.mjd, stop.mjd, 1000), format="mjd")
        eph = utils.get_body_ephemeris(times, body_id="501", step="1m")

        # Central longitude
        assert np.allclose(eph_jpl["PDObsLon"][0], eph["theta"][0].value)

        # Inclination
        if eph_jpl["PDObsLat"][0] > 0:
            assert eph["inc"][0].value < 90
        else:
            assert eph["inc"][0].value > 90

        # Test position of the sun w.r.t target body
        # subsolar point is west of central point
        if (eph_jpl["PDSunLon"][0] - eph_jpl["PDObsLon"][0]) > 0:
            assert eph["xs"][0].value < 0.0
        # subsolar point is east of central point
        else:
            assert eph["xs"][0].value > 0.0

        # subsolar point is above the central latitude
        if (eph_jpl["PDSunLat"][0] - eph_jpl["PDObsLat"][0]) > 0:
            assert eph["ys"][0].value > 0.0
        # subsolar point is below the central latitude
        else:
            assert eph["ys"][0].value < 0.0


def test_get_occultation_lattitude():
    # Test that the sign of the latitude is correct
    x_io = -39.0
    y_io = 11.0
    re = 40.0

    lat = utils.get_occultation_latitude(x_io, y_io, re)

    if lat < 0.0:
        raise ValueError(
            "Occultation latitude should be positive.", "Something is wrong."
        )

    lat = utils.get_occultation_latitude(x_io, -y_io, re)

    if lat > 0.0:
        raise ValueError(
            "Occultation latitude should be negative.", "Something is wrong."
        )
