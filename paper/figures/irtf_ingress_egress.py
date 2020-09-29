import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import theano
import theano.tensor as tt
import pymc3 as pm
import starry
import exoplanet as xo
from astropy import units as u
from astropy.table import Table
from astropy.timeseries import TimeSeries

from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator

from volcano.utils import *

np.random.seed(42)
starry.config.lazy = True


# Load light curves
lc_in_n = np.load("irtf_ingress.npy")
lc_eg_n = np.load("irtf_egress.npy")

lc_in = TimeSeries(
    time=Time(lc_in_n[:, 0], format="mjd"),
    data=Table(
        lc_in_n[:, 1:] * u.GW / u.um / u.sr, names=["flux", "flux_err"]
    ),
)
lc_eg = TimeSeries(
    time=Time(lc_eg_n[:, 0], format="mjd"),
    data=Table(
        lc_eg_n[:, 1:] * u.GW / u.um / u.sr, names=["flux", "flux_err"]
    ),
)

# Compute ephemeris
eph_list_io = []
eph_list_jup = []

for lc in (lc_in, lc_eg):
    times = lc.time

    eph_io = get_body_ephemeris(
        times, body_id="501", step="1m", return_orientation=True
    )
    eph_jup = get_body_ephemeris(
        times, body_id="599", step="1m", return_orientation=True
    )

    eph_list_io.append(eph_io)
    eph_list_jup.append(eph_jup)

eph_io_in = eph_list_io[0]
eph_jup_in = eph_list_jup[0]
eph_io_eg = eph_list_io[1]
eph_jup_eg = eph_list_jup[1]

t_in = (lc_in.time.mjd - lc_in.time.mjd[0]) * 24 * 60
t_eg = (lc_eg.time.mjd - lc_eg.time.mjd[0]) * 24 * 60

f_obs_in = lc_in["flux"].value
f_err_in = lc_in["flux_err"].value
f_obs_eg = lc_eg["flux"].value
f_err_eg = lc_eg["flux_err"].value

f_obs = np.concatenate([f_obs_in, f_obs_eg])
f_err = np.concatenate([f_err_in, f_err_eg])


def get_pos_rot(eph_io, eph_jup, method=""):
    """
    Get sky position of Io relative to Jupiter rotated such that the
    obliquity of Jupiter is zero.
    """
    # Get occultor position
    obl = eph_io["obl"]
    inc = np.mean(eph_jup["inc"])
    theta = np.array(eph_io["theta"])

    xo_unrot, yo_unrot, zo, ro = get_occultor_position_and_radius(
        eph_io, eph_jup, occultor_is_jupiter=True, method=method
    )

    # Rotate to coordinate system where the obliquity of Io is 0
    theta_rot = -obl.to(u.rad)
    xo_rot, yo_rot = rotate_vectors(xo_unrot, yo_unrot, theta_rot)

    return xo_rot, yo_rot, ro


xo_in, yo_in, ro_in = get_pos_rot(eph_io_in, eph_jup_in)
xo_eg, yo_eg, ro_eg = get_pos_rot(eph_io_eg, eph_jup_eg)

# Phase
theta_in = eph_io_in["theta"].value
theta_eg = eph_io_eg["theta"].value

# Fit single map model with different map amplitudes for ingress and egress
ydeg_inf = 25
map = starry.Map(ydeg_inf)
lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
npix = Y2P.shape[0]

# Evalute MAP model on denser grid
xo_dense_in = np.linspace(xo_in[0], xo_in[-1], 200)
yo_dense_in = np.linspace(yo_in[0], yo_in[-1], 200)
theta_dense_in = np.linspace(theta_in[0], theta_in[-1], 200)

xo_dense_eg = np.linspace(xo_eg[0], xo_eg[-1], 200)
yo_dense_eg = np.linspace(yo_eg[0], yo_eg[-1], 200)
theta_dense_eg = np.linspace(theta_eg[0], theta_eg[-1], 200)

t_dense_in = np.linspace(t_in[0], t_in[-1], 200)
t_dense_eg = np.linspace(t_eg[0], t_eg[-1], 200)


def get_S(ydeg, sigma=0.1):
    l = np.concatenate([np.repeat(l, 2 * l + 1) for l in range(ydeg + 1)])
    s = np.exp(-0.5 * l * (l + 1) * sigma ** 2)
    S = np.diag(s)
    return S


with pm.Model() as model:
    map = starry.Map(ydeg_inf)

    p = pm.Exponential(
        "p", 1 / 10.0, shape=(npix,), testval=0.1 * np.random.rand(npix)
    )
    x = tt.dot(P2Y, p)

    # Run the smoothing filter
    S = get_S(ydeg_inf, 2 / ydeg_inf)
    x_s = tt.dot(S, x[:, None]).flatten()

    map.amp = x_s[0]
    map[1:, :] = x_s[1:] / x_s[0]

    pm.Deterministic("map_in_amp", map.amp)
    pm.Deterministic("map_in_y1", map[1:, :])

    # Flux offset
    ln_flux_offset = pm.Normal(
        "ln_flux_offset", 0.0, 4, shape=(2,), testval=-2 * np.ones(2)
    )

    # Compute flux
    flux_in = map.flux(
        xo=theano.shared(xo_in),
        yo=theano.shared(yo_in),
        ro=ro_in,
        theta=theano.shared(theta_in),
    ) + tt.exp(ln_flux_offset[0])

    # Dense grid
    pm.Deterministic(
        "flux_dense_in",
        map.flux(
            xo=theano.shared(xo_dense_in),
            yo=theano.shared(yo_dense_in),
            ro=ro_in,
            theta=theano.shared(theta_dense_in),
        )
        + tt.exp(ln_flux_offset[0]),
    )

    # Same coefficients for the second map except rescale amplitude
    amp_eg = pm.Normal("amp_eg", 1.0, 0.1, testval=1.0)
    map.amp *= amp_eg

    pm.Deterministic("map_eg_amp", map.amp)
    pm.Deterministic("map_eg_y1", map[1:, :])

    flux_eg = map.flux(
        xo=theano.shared(xo_eg),
        yo=theano.shared(yo_eg),
        ro=ro_eg,
        theta=theano.shared(theta_eg),
    ) + tt.exp(ln_flux_offset[1])

    # Dense grid
    pm.Deterministic(
        "flux_dense_eg",
        map.flux(
            xo=theano.shared(xo_dense_eg),
            yo=theano.shared(yo_dense_eg),
            ro=ro_eg,
            theta=theano.shared(theta_dense_eg),
        )
        + tt.exp(ln_flux_offset[1]),
    )

    pm.Deterministic("flux_pred_in", flux_in)
    pm.Deterministic("flux_pred_eg", flux_eg)
    flux = tt.concatenate([flux_in, flux_eg])

    sig = pm.Normal("sig", f_err[0], 0.05 * f_err[0], testval=f_err[0])
    pm.Normal("obs", mu=flux, sd=sig * tt.ones(len(f_obs)), observed=f_obs)

with model:
    soln = xo.optimize(options=dict(maxiter=99999))

# Compute residuals
res_in = lc_in["flux"].value - soln["flux_pred_in"]
res_in = res_in / np.std(res_in)

res_eg = lc_eg["flux"].value - soln["flux_pred_eg"]
res_eg = res_eg / np.std(res_eg)

# Initialize maps
map = starry.Map(ydeg_inf)
map.amp = soln["map_in_amp"]
map[1:, :] = soln["map_in_y1"]

map_eg = starry.Map(ydeg_inf)
map_eg.amp = soln["map_eg_amp"]
map_eg[1:, :] = soln["map_eg_y1"]

# Set up the plot
resol = 150
nim = 8
cmap_norm = colors.Normalize(vmin=-0.5, vmax=900)

fig = plt.figure(figsize=(9, 7))
fig.subplots_adjust(wspace=0.0)

heights = [2, 3, 1]
gs0 = fig.add_gridspec(
    nrows=1, ncols=2 * nim, bottom=0.65, left=0.05, right=0.98,
)
gs1 = fig.add_gridspec(
    nrows=3, ncols=nim, height_ratios=heights, top=0.67, left=0.05, right=0.50
)
gs2 = fig.add_gridspec(
    nrows=3, ncols=nim, height_ratios=heights, top=0.67, left=0.53, right=0.98
)

# Maps
ax_map_in = fig.add_subplot(gs0[0, :nim])
ax_map_eg = fig.add_subplot(gs0[0, nim:])
map.show(
    ax=ax_map_in,
    projection="molleweide",
    colorbar=False,
    norm=cmap_norm,
    res=resol,
)
map_eg.show(
    ax=ax_map_eg,
    projection="molleweide",
    colorbar=True,
    norm=cmap_norm,
    res=resol,
)
ax_map_in.set_title("Inferred map\n ingress")
ax_map_eg.set_title("Inferred map\n egress")

# Minimaps
ax_im = [
    [fig.add_subplot(gs1[0, i]) for i in range(nim)],
    [fig.add_subplot(gs2[0, i]) for i in range(nim)],
]

# Light curves
ax_lc = [fig.add_subplot(gs1[1, :]), fig.add_subplot(gs2[1, :])]

# Residuals
ax_res = [fig.add_subplot(gs1[2, :]), fig.add_subplot(gs2[2, :])]

# Plot minimaps
xo_im_in = np.linspace(xo_in[0], xo_in[-1], nim)
yo_im_in = np.linspace(yo_in[0], yo_in[-1], nim)
xo_im_eg = np.linspace(xo_eg[0], xo_eg[-1], nim)
yo_im_eg = np.linspace(yo_eg[0], yo_eg[-1], nim)
xo_im = [xo_im_in, xo_im_eg]
yo_im = [yo_im_in, yo_im_eg]

for j, ro in enumerate((ro_in, ro_eg)):
    a = ax_im[j]
    for n in range(nim):
        # Show the image
        map.show(ax=a[n], res=resol, grid=False, norm=cmap_norm)

        # Outline
        x = np.linspace(-1, 1, 1000)
        y = np.sqrt(1 - x ** 2)
        f = 0.98
        a[n].plot(f * x, f * y, "k-", lw=0.5, zorder=0)
        a[n].plot(f * x, -f * y, "k-", lw=0.5, zorder=0)

        # Occultor
        x = np.linspace(-1.5, xo_im[j][n] + ro - 1e-5, resol)
        y = np.sqrt(ro ** 2 - (x - xo_im[j][n]) ** 2)
        a[n].fill_between(
            x,
            yo_im[j][n] - y,
            yo_im[j][n] + y,
            fc="w",
            zorder=1,
            clip_on=True,
            ec="k",
            lw=0.5,
        )
        a[n].axis("off")
        a[n].set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        a[n].set_rasterization_zorder(0)

# Plot ingress
ax_lc[0].errorbar(  # Data
    t_in,
    f_obs_in,
    f_err_in,
    color="black",
    fmt=".",
    ecolor="black",
    alpha=0.4,
)

ax_lc[0].plot(t_dense_in, soln["flux_dense_in"], "C1-")  # Model

# Residuals
ax_res[0].errorbar(
    t_in, res_in, f_err_in, color="black", fmt=".", ecolor="black", alpha=0.4,
)

# Plot egress
ax_lc[1].errorbar(
    t_eg,
    f_obs_eg,
    f_err_eg,
    color="black",
    fmt=".",
    ecolor="black",
    alpha=0.4,
)

ax_lc[1].plot(t_dense_eg, soln["flux_dense_eg"], "C1-")

ax_res[1].errorbar(
    t_eg, res_eg, f_err_eg, color="black", fmt=".", ecolor="black", alpha=0.4,
)


# Make broken axis
for ax in (ax_lc, ax_res):
    ax[0].spines["right"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].tick_params(axis="y", colors=(0, 0, 0, 0))

    d = 0.01
    kwargs = dict(transform=ax[0].transAxes, color="k", clip_on=False)
    ax[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    ax[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax[1].transAxes)
    ax[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax[1].plot((-d, +d), (-d, +d), **kwargs)

#  Ticks
for a in (ax_lc[0], ax_res[0]):
    a.set_xticks(np.arange(0, 4.5, 0.5))
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())

for a in (ax_lc[1], ax_res[1]):
    a.set_xticks(np.arange(0, 6.0, 0.5))
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_ticklabels([])
    a.yaxis.set_ticklabels([])
    a.set_ylabel("")
    a.set_ylabel("")

for a in ax_lc:
    a.set_ylim(-2, 52)
    a.set_xticklabels([])
    a.set_yticks(np.arange(0, 60, 10))

for a in ax_lc + ax_res:
    a.grid()

for a in ax_res:
    a.set_yticks([-3.5, 0.0, 3.5])

ax_res[0].set_xlabel(f"Minutes from {lc_in.time[0].iso[:-4]}")
ax_res[1].set_xlabel(f"Minutes from {lc_eg.time[0].iso[:-4]}")

# Set common labels
# fig.text(0.5, 0.04, "Duration [minutes]", ha="center", va="center")
ax_lc[0].set_ylabel("Flux [GW/sr/um]")
ax_res[0].set_ylabel("Residuals\n (norm.)")

# Save plot
fig.savefig("irtf_ingress_egress.pdf", bbox_inches="tight", dpi=500)

# Plot inferred map over geological map of Io
combined_mosaic = Image.open("Io_SSI_VGR_bw_Equi_Clon180_8bit.tif")

# The loaded map is [0,360] while Starry expects [-180, 180]
shifted_map = np.roll(combined_mosaic, int(11445 / 2), axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
map.show(projection="rectangular", ax=ax, colorbar=False, res=resol)

ax.imshow(shifted_map, extent=(-180, 180, -90, 90), alpha=0.7, cmap="gray")
ax.set_yticks(np.arange(-15, 60, 15))
ax.set_xticks(np.arange(15, 105, 15))
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.set_xlim(15, 90)
ax.set_ylim(-15, 45)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Save plot
fig.savefig("irtf_ingress_egress_loki.pdf", bbox_inches="tight", dpi=500)
