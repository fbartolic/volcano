import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pickle as pkl

import starry

from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
from matplotlib.lines import Line2D

from scipy.integrate import dblquad

from volcano.utils import get_median_map

np.random.seed(42)
starry.config.lazy = False


def integrand(theta, phi, map):
    lat = theta * 180 / np.pi - 90
    lon = phi * 180 / np.pi

    return map.intensity(lat=lat, lon=lon) * np.sin(theta)


def get_power_emitted(map, lat_range, lon_range):
    phi_min = lon_range[0] * np.pi / 180
    phi_max = lon_range[1] * np.pi / 180
    theta_min = lat_range[0] * np.pi / 180 + np.pi / 2
    theta_max = lat_range[1] * np.pi / 180 + np.pi / 2

    res, _ = dblquad(
        integrand,
        phi_min,
        phi_max,
        lambda x: theta_min,
        lambda x: theta_max,
        epsabs=1e-4,
        epsrel=1e-4,
        args=(map,),
    )
    return res


def find_centroid(lats, lons, img, bounds):
    mask_lat = np.logical_and(lats > bounds[0][0], lats < bounds[0][1])
    mask_lon = np.logical_and(lons > bounds[1][0], lons < bounds[1][1])

    img_bounded = img[mask_lat][:, mask_lon]

    I_min = np.nanpercentile(img_bounded, [90])
    mask = img_bounded > I_min

    lon_grid, lat_grid = np.meshgrid(lons[mask_lon], lats[mask_lat])
    return np.mean(lon_grid[mask]), np.mean(lat_grid[mask])


def get_spot_position_and_power(samples, bounds=None, nsamples=100):
    """
    For each sample find local maximum to get spot position,
    then compute emitted power from the spot.
    """
    lat_list = []
    lon_list = []
    power_list_in = []
    power_list_eg = []

    map = starry.Map(20)

    for n in np.random.randint(len(samples), size=nsamples):
        x = samples["x_in"][n]
        map[1:, :] = x[1:] / x[0]
        map.amp = x[0]

        # Find maximum
        #        map[1:, :] = -map[1:, :]
        #        lat, lon, _ = map.minimize(oversample=2, ntries=2, bounds=bounds)
        #        map[1:, :] = -map[1:, :]
        img = map.render(res=400, projection="rect")
        lats, lons = map.get_latlon_grid(res=400, projection="rect")
        lon, lat = find_centroid(lats[:, 0], lons[0, :], img, bounds)

        lat_list.append(lat)
        lon_list.append(lon)

        # Get total emitted power in circle around inferred spot center
        power = get_power_emitted(
            map, (lat - 15, lat + 15), (lon - 15, lon + 15)
        )
        power_list_in.append(power)
        power_list_eg.append(samples["amp_eg"] * power)

    return (
        np.array(lat_list),
        np.array(lon_list),
        np.array(power_list_in),
        np.array(power_list_eg),
    )


def print_percentiles(samples, varname):
    mcmc = np.percentile(samples, [16, 50, 84])
    q = np.diff(mcmc)
    print(f"{varname}: {mcmc[1]:.3f} {q[0]:.3f} {q[1]:.3f}")


# Plots for the the 1998 pair of light curves
with open("../../data/irtf_processed/lc_1998-08-27.pkl", "rb") as handle:
    lc_in = pkl.load(handle)

with open("../../data/irtf_processed/lc_1998-11-29.pkl", "rb") as handle:
    lc_eg = pkl.load(handle)

with open("scripts/irtf_1998_samples.pkl", "rb") as handle:
    samples = pkl.load(handle)


# Plot hotspots
(
    spot1_lat,
    spot1_lon,
    spot1_power_in,
    spot1_power_eg,
) = get_spot_position_and_power(samples, bounds=((-20, 50), (20, 100)))
(
    spot2_lat,
    spot2_lon,
    spot2_power_in,
    spot2_power_eg,
) = get_spot_position_and_power(samples, bounds=((-50, 20), (-60, 0)))

print("1998 spot1 parameters:")
print_percentiles(spot1_lat, "spot latitude")
print_percentiles(spot1_lon, "spot longitude")
print_percentiles(spot1_power_in, "spot power ingress")
print_percentiles(spot1_power_eg, "spot power egress")

print("1998 spot2 parameters:")
print_percentiles(spot2_lat, "spot latitude")
print_percentiles(spot2_lon, "spot longitude")
print_percentiles(spot2_power_in, "spot power ingress")
print_percentiles(spot2_power_eg, "spot power egress")

# Load Galileo map
combined_mosaic = Image.open("Io_SSI_VGR_bw_Equi_Clon180_8bit.tif")
# Longitude is defined on [0,360] while Starry expects [-180, 180]
shifted_map = np.roll(combined_mosaic, int(11445 / 2), axis=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
median_map_rect_in = get_median_map(
    20, samples["x_in"], projection="rect", nsamples=50, resol=400
)

map = starry.Map(20)
lat_grid, lon_grid = map.get_latlon_grid(res=400, projection="rect")

# Bright spot
ax[0].contour(lon_grid, lat_grid, median_map_rect_in, cmap="Oranges")
ax[0].imshow(shifted_map, extent=(-180, 180, -90, 90), alpha=0.7, cmap="gray")
ax[0].set(
    xlim=(35, 65),
    ylim=(0, 30),
)


def get_errobar(samples):
    mcmc = np.percentile(samples, [16, 50, 84])
    q = np.diff(mcmc)
    return np.array([q[0], q[1]])[:, None]


ax[0].errorbar(
    np.median(spot1_lon),
    np.median(spot1_lat),
    yerr=get_errobar(spot1_lat),
    xerr=get_errobar(spot1_lon),
    color="white",
)

# Dim spot
ax[1].contour(
    lon_grid,
    lat_grid,
    median_map_rect_in,
    cmap="Oranges",
    levels=np.arange(20, 60, 10),
)
ax[1].imshow(shifted_map, extent=(-180, 180, -90, 90), alpha=0.7, cmap="gray")
ax[1].set(
    xlim=(-55, -25),
    ylim=(-30, 0),
)

ax[1].errorbar(
    np.median(spot2_lon),
    np.median(spot2_lat),
    yerr=get_errobar(spot2_lat),
    xerr=get_errobar(spot2_lon),
    color="white",
)

for a in ax.flatten():
    a.set_aspect(1)
    a.set_xlabel("East longitude [deg]")

fig.text(0.05, 0.5, "Latitude [deg]", va="center", rotation="vertical")

for a in ax:
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())

fig.savefig("irtf_1998_spots.pdf", bbox_inches="tight", dpi=500)

# Plots for the 2017 pair of light curves
with open("../../data/irtf_processed/lc_2017-03-31.pkl", "rb") as handle:
    lc_in = pkl.load(handle)

with open("../../data/irtf_processed/lc_2017-05-11.pkl", "rb") as handle:
    lc_eg = pkl.load(handle)

with open("scripts/irtf_2017_samples.pkl", "rb") as handle:
    samples2 = pkl.load(handle)

(
    spot1_lat,
    spot1_lon,
    spot1_power_in,
    spot1_power_eg,
) = get_spot_position_and_power(samples2, bounds=((-20, 50), (20, 100)))
(
    spot2_lat,
    spot2_lon,
    spot2_power_in,
    spot2_power_eg,
) = get_spot_position_and_power(samples2, bounds=((-30, 30), (-70, -15)))

print("2017 spot1 parameters:")
print_percentiles(spot1_lat, "spot latitude")
print_percentiles(spot1_lon, "spot longitude")
print_percentiles(spot1_power_in, "spot power ingress")
print_percentiles(spot1_power_eg, "spot power egress")

print("2017 spot2 parameters:")
print_percentiles(spot2_lat, "spot latitude")
print_percentiles(spot2_lon, "spot longitude")
print_percentiles(spot2_power_in, "spot power ingress")
print_percentiles(spot2_power_eg, "spot power egress")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
median_map_rect_in = get_median_map(
    20, samples2["x_in"], projection="rect", nsamples=50, resol=400
)

# Bright spot
ax[0].contour(lon_grid, lat_grid, median_map_rect_in, cmap="Oranges")
ax[0].imshow(shifted_map, extent=(-180, 180, -90, 90), alpha=0.7, cmap="gray")
ax[0].set(
    xlim=(35, 65),
    ylim=(0, 30),
)

ax[0].errorbar(
    np.median(spot1_lon),
    np.median(spot1_lat),
    yerr=get_errobar(spot1_lat),
    xerr=get_errobar(spot1_lon),
    color="white",
)

# Dim spot
ax[1].contour(
    lon_grid,
    lat_grid,
    median_map_rect_in,
    cmap="Oranges",
    levels=np.arange(20, 60, 10),
)
ax[1].imshow(shifted_map, extent=(-180, 180, -90, 90), alpha=0.7, cmap="gray")
ax[1].set(
    xlim=(-55, -25),
    ylim=(-30, 0),
)

ax[1].errorbar(
    np.median(spot2_lon),
    np.median(spot2_lat),
    yerr=get_errobar(spot2_lat),
    xerr=get_errobar(spot2_lon),
    color="white",
)

for a in ax.flatten():
    a.set_aspect(1)
    a.set_xlabel("East longitude [deg]")

fig.text(0.05, 0.5, "Latitude [deg]", va="center", rotation="vertical")

for a in ax:
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())

fig.savefig("irtf_2017_spots.pdf", bbox_inches="tight", dpi=500)
