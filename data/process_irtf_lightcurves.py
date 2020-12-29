import numpy as np
import os
import xml.etree.ElementTree as ET
import sys
import pickle as pkl

from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.table import vstack
from astropy.table import Table
from astropy.stats import sigma_clip, mad_std

from scipy.signal import savgol_filter

np.random.seed(42)


def get_data_from_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    namespaces = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}

    description = root.findall(".//pds:description", namespaces=namespaces)[
        0
    ].text
    start_date = root.findall(".//pds:start_date_time", namespaces=namespaces)[
        0
    ].text
    end_date = root.findall(".//pds:stop_date_time", namespaces=namespaces)[
        0
    ].text
    instrument = root.findall(
        ".//pds:Observing_System_Component", namespaces=namespaces
    )[0][0].text
    facility = root.findall(
        ".//pds:Observing_System_Component", namespaces=namespaces
    )[1][0].text

    dic = {
        "description": description,
        "start_date": start_date,
        "end_date": end_date,
        "instrument": instrument,
        "facility": facility,
    }
    return dic


directory = "irtf_raw"

lcs = []
i = 0
for filename in os.listdir(directory):
    if filename.endswith(".xml"):
        metadata = get_data_from_xml(os.path.join(directory, filename))
        data = np.genfromtxt(
            os.path.join(directory, filename[:-4] + "tab.txt")
        )

        # Remove the zeros at the beginning of some files
        mask = data[:, 0] == 0
        data = data[~mask]

        start_time = Time(
            metadata["start_date"][:-1], format="isot", scale="utc"
        ).to_value("mjd", "long")
        times = start_time + data[:, 0] / 24.0
        times = Time(times, format="mjd")

        # Convert to astropy.TimeSeries
        ts = TimeSeries(time=times)
        ts["flux"] = data[:, 2] * u.GW / u.um / u.sr
        ts["phase"] = data[:, 1]
        ts.meta = metadata

        lcs.append(ts)
        i += 1

# Remove some points at the beginning of two events for which the times
# are all messed up
for i, lc in enumerate(lcs):
    ts = lc.time.mjd - lc.time.mjd[0]
    mask = ts == 0

    if mask.sum() > 1.0:
        lcs[i] = lc[~mask]


def clip_outliers(f, sigma=5):
    """Filter with Savitzsky Golay and do sigma clipping."""
    round_up_to_odd = lambda n: int(np.ceil(n) // 2 * 2 + 1)
    win_len = round_up_to_odd(0.05 * len(f))

    if win_len < 5:
        win_len = 5

    f_smooth = savgol_filter(f, win_len, 3, mode="mirror")
    res = f - f_smooth

    clipped = sigma_clip(
        res,
        sigma=5,
        maxiters=None,
        cenfunc="median",
        stdfunc=mad_std,
        masked=True,
        copy=False,
    )
    return clipped.mask, clipped


lcs_processed = []

# Remove outliers, estimate errorbars
for idx in range(len(lcs)):
    f = lcs[idx]["flux"].value
    mask, res = clip_outliers(f)

    # Estimate errorbars
    lcs[idx]["flux_err"] = np.ones_like(f) * np.std(res) * u.GW / u.um / u.sr
    lcs_processed.append(lcs[idx][~mask])

# Save processed light curves to disk
output_dir = "irtf_processed"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for lc in lcs_processed:
    date = lc["time"][0].isot[:10]
    filename = os.path.join(output_dir, f"lc_{date}.pkl")
    with open(filename, "wb") as handle:
        pkl.dump(lc, handle)
