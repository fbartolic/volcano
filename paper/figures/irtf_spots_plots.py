import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pickle as pkl

import starry

from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


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


def get_contour_levels(lats, lons, img, bounds, percentiles):
    mask_lat = np.logical_and(lats > bounds[0][0], lats < bounds[0][1])
    mask_lon = np.logical_and(lons > bounds[1][0], lons < bounds[1][1])

    img_bounded = img[mask_lat][:, mask_lon]

    # Define background as all pix < 0.5 * max intensity
    I_min = 0.5 * np.nanmax(img_bounded)
    mask = img_bounded > I_min

    lon_grid, lat_grid = np.meshgrid(lons[mask_lon], lats[mask_lat])
    return np.nanpercentile(img_bounded[mask].reshape(-1), percentiles)


def get_spot_position_and_power(samples, bounds=None, nsamples=300):
    """
    For each sample find local maximum to get spot position,
    then compute emitted power from the spot.
    """
    lat_list = []
    lon_list = []
    power_list_in = []
    power_list_eg = []

    map = starry.Map(20)

    for n in np.random.randint(len(samples["x_in"]), size=nsamples):
        x = samples["x_in"][n]
        map[1:, :] = x[1:] / x[0]
        map.amp = x[0]

        # Find maximum
        map[1:, :] = -map[1:, :]
        lat, lon, _ = map.minimize(oversample=2, ntries=2, bounds=bounds)
        map[1:, :] = -map[1:, :]
        #        img = map.render(res=400, projection="rect")
        #        lats, lons = map.get_latlon_grid(res=400, projection="rect")
        #        lon, lat = find_centroid(lats[:, 0], lons[0, :], img, bounds)

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


def get_errobar(samples):
    mcmc = np.percentile(samples, [16, 50, 84])
    q = np.diff(mcmc)
    return np.array([q[0], q[1]])[:, None]


# Load Galileo map
combined_mosaic = Image.open("Io_color_merged_clon180.tif")
# Longitude is defined on [0,360] while Starry expects [-180, 180]
shifted_map = np.roll(combined_mosaic, int(11445 / 2), axis=1)


map = starry.Map(20)
lat_grid, lon_grid = map.get_latlon_grid(res=400, projection="rect")


def make_plot(
    samples, spot1_bounds, spot2_bounds, xlim1, ylim1, xlim2, ylim2, year
):
    map = starry.Map(20)
    lat_grid, lon_grid = map.get_latlon_grid(res=400, projection="rect")

    # Estimate spot positions and emitted powe
    (
        spot1_lat,
        spot1_lon,
        spot1_power_in,
        spot1_power_eg,
    ) = get_spot_position_and_power(samples, bounds=spot1_bounds)
    (
        spot2_lat,
        spot2_lon,
        spot2_power_in,
        spot2_power_eg,
    ) = get_spot_position_and_power(samples, bounds=spot2_bounds)

    # Print spot parameters
    print(f"{year} spot1 parameters:")
    print_percentiles(spot1_lat, "spot latitude")
    print_percentiles(spot1_lon, "spot longitude")
    print_percentiles(spot1_power_in, "spot power ingress")
    print_percentiles(spot1_power_eg, "spot power egress")

    print(f"{year} spot2 parameters:")
    print_percentiles(spot2_lat, "spot latitude")
    print_percentiles(spot2_lon, "spot longitude")
    print_percentiles(spot2_power_in, "spot power ingress")
    print_percentiles(spot2_power_eg, "spot power egress")

    median_map_rect_in = get_median_map(
        20, samples["x_in"], projection="rect", nsamples=50, resol=400
    )

    lvls_spot1 = get_contour_levels(
        lat_grid[:, 0],
        lon_grid[0, :],
        median_map_rect_in,
        spot1_bounds,
        [5, 50, 95],
    )
    lvls_spot2 = get_contour_levels(
        lat_grid[:, 0],
        lon_grid[0, :],
        median_map_rect_in,
        spot2_bounds,
        [5, 50, 95],
    )

    # Make plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Bright spot
    ax[0].contour(
        lon_grid,
        lat_grid,
        median_map_rect_in,
        cmap="OrRd",
        levels=lvls_spot1,
    )
    ax[0].imshow(shifted_map, extent=(-180, 180, -90, 90))
    ax[0].set(
        xlim=xlim1,
        ylim=ylim1,
    )

    ax[0].errorbar(
        np.median(spot1_lon),
        np.median(spot1_lat),
        yerr=get_errobar(spot1_lat)[::-1],  # (-, +)
        xerr=get_errobar(spot1_lon)[::-1],
        color="white",
    )

    # Dim spot
    ax[1].contour(
        lon_grid,
        lat_grid,
        median_map_rect_in,
        cmap="OrRd",
        levels=lvls_spot2,
    )
    ax[1].imshow(shifted_map, extent=(-180, 180, -90, 90))
    ax[1].set(
        xlim=xlim2,
        ylim=ylim2,
    )

    ax[1].errorbar(
        np.median(spot2_lon),
        np.median(spot2_lat),
        yerr=get_errobar(spot2_lat)[::-1],
        xerr=get_errobar(spot2_lon)[::-1],
        color="white",
    )

    # Scale indicator
    scale = 3.1455  # 100km

    fontprops = fm.FontProperties(size=8)
    scalebar1 = AnchoredSizeBar(
        ax[0].transData,
        scale,
        "100 km",
        "lower right",
        pad=0.8,
        color="white",
        frameon=False,
        size_vertical=0.15,
        fontproperties=fontprops,
    )

    ax[0].add_artist(scalebar1)

    scalebar2 = AnchoredSizeBar(
        ax[1].transData,
        scale,
        "100 km",
        "lower right",
        pad=0.8,
        color="white",
        frameon=False,
        size_vertical=0.15,
        fontproperties=fontprops,
    )

    ax[1].add_artist(scalebar2)

    for a in ax.flatten():
        a.set_aspect(1)
        a.set_xlabel("East longitude [deg]")

    fig.text(0.04, 0.5, "Latitude [deg]", va="center", rotation="vertical")
    fig.text(
        0.5,
        0.9,
        f"Inferred hot spots from the {year} pair of occultations",
        ha="center",
    )

    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())

    fig.savefig(f"irtf_{year}_spots.pdf", bbox_inches="tight", dpi=500)


# Plots for the the 1998 pair of light curves
with open("scripts/irtf_1998_samples.pkl", "rb") as handle:
    samples = pkl.load(handle)


# Get levels for the contour plot
spot1_bounds = ((-20, 50), (20, 100))
spot2_bounds = ((-40, 20), (-60, -20))

make_plot(
    samples,
    spot1_bounds,
    spot2_bounds,
    (40, 40 + 25),
    (2, 2 + 25),
    (-49, -49 + 25),
    (-27, -27 + 25),
    "1998",
)


# Plots for the 2017 pair of light curves
with open("scripts/irtf_2017_samples.pkl", "rb") as handle:
    samples2 = pkl.load(handle)

# Get levels for the contour plot
spot1_bounds = ((-20, 50), (20, 100))
spot2_bounds = ((-40, 20), (-60, -20))

make_plot(
    samples2,
    spot1_bounds,
    spot2_bounds,
    (40, 40 + 25),
    (2, 2 + 25),
    (-50, -50 + 20),
    (-15, -15 + 20),
    "2017",
)