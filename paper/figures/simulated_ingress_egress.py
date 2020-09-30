import numpy as np
from matplotlib import pyplot as plt

import theano
import theano.tensor as tt
import pymc3 as pm
import starry
import exoplanet as xo
from scipy import optimize

from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator

from volcano.utils import *

np.random.seed(42)
starry.config.lazy = True


def get_S(ydeg, sigma=0.1):
    l = np.concatenate([np.repeat(l, 2 * l + 1) for l in range(ydeg + 1)])
    s = np.exp(-0.5 * l * (l + 1) * sigma ** 2)
    S = np.diag(s)
    return S


xo_eg = np.linspace(37.15, 39.43, 150)
yo_eg = np.linspace(-8.284, -8.27, 150)

xo_in = np.linspace(-39.5, -37.3, 150)
yo_in = np.linspace(-7.714, -7.726, 150)

xo_com = np.concatenate([xo_in, xo_eg])
yo_com = np.concatenate([yo_in, yo_eg])

ro = 39.1

ydeg_true = 25
map_true = starry.Map(ydeg_true)
spot_lon = 360 - 309
spot_lat = 13.0
spot_ang_dim = 5 * np.pi / 180
spot_sigma = 1 - np.cos(spot_ang_dim / 2)
map_true.add_spot(
    amp=1.0, sigma=spot_sigma, lat=spot_lat, lon=spot_lon, relative=False
)
map_true.amp = 20

# Smooth the true map
sigma_s = 2 / ydeg_true
S_true = get_S(ydeg_true, sigma_s)
x = (map_true.amp * map_true.y).eval()
x_smooth = (S_true @ x[:, None]).reshape(-1)
map_true[:, :] = x_smooth / x_smooth[0]
map_true.amp = x_smooth[0]

# Generate mock ingress and egress light curves
f_true_in = map_true.flux(ro=ro, xo=xo_in, yo=yo_in).eval()
f_true_eg = map_true.flux(ro=ro, xo=xo_eg, yo=yo_eg).eval()

f_err_in = 0.02 * np.max(f_true_in)
f_err_eg = 0.02 * np.max(f_true_eg)

f_obs_in = f_true_in + np.random.normal(0, f_err_in, len(f_true_in))
f_obs_eg = f_true_eg + np.random.normal(0, f_err_eg, len(f_true_eg))
f_obs = np.concatenate([f_obs_in, f_obs_eg])

# Set up model
ydeg_inf = 25
map = starry.Map(ydeg_inf)
lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
npix = Y2P.shape[0]

# Evalute MAP model on denser grid
xo_in_dense = np.linspace(xo_in[0], xo_in[-1], 200)
yo_in_dense = np.linspace(yo_in[0], yo_in[-1], 200)
xo_eg_dense = np.linspace(xo_eg[0], xo_eg[-1], 200)
yo_eg_dense = np.linspace(yo_eg[0], yo_eg[-1], 200)

# Compute design matrix
map = starry.Map(ydeg_inf)
A_in = theano.shared(map.design_matrix(xo=xo_in, yo=yo_in, ro=ro).eval())
A_eg = theano.shared(map.design_matrix(xo=xo_eg, yo=yo_eg, ro=ro).eval())
A_in_dense = theano.shared(
    map.design_matrix(xo=xo_in_dense, yo=yo_in_dense, ro=ro).eval()
)
A_eg_dense = theano.shared(
    map.design_matrix(xo=xo_eg_dense, yo=yo_eg_dense, ro=ro).eval()
)


with pm.Model() as model:
    p = pm.Exponential("p", 1 / 10.0, shape=(npix,))
    x = tt.dot(P2Y, p)

    # Run the smoothing filter
    S = get_S(ydeg_inf, 2 / ydeg_inf)
    x_s = tt.dot(S, x[:, None]).flatten()

    pm.Deterministic("amp", x_s[0])
    pm.Deterministic("y1", x_s[1:] / x_s[0])

    # Compute flux
    flux_in = tt.dot(A_in, x_s[:, None]).flatten()
    flux_eg = tt.dot(A_eg, x_s[:, None]).flatten()
    pm.Deterministic("flux_pred_in", flux_in)
    pm.Deterministic("flux_pred_eg", flux_eg)
    flux = tt.concatenate([flux_in, flux_eg])

    # Dense grid
    flux_in_dense = tt.dot(A_in_dense, x_s[:, None]).flatten()
    flux_eg_dense = tt.dot(A_eg_dense, x_s[:, None]).flatten()
    pm.Deterministic("flux_in_dense", flux_in_dense)
    pm.Deterministic("flux_eg_dense", flux_eg_dense)
    flux_dense = tt.concatenate([flux_in_dense, flux_eg_dense])

    pm.Normal(
        "obs_in",
        mu=flux_in,
        sd=f_err_in * np.ones_like(f_obs_in),
        observed=f_obs_in,
    )
    pm.Normal(
        "obs_eg",
        mu=flux_eg,
        sd=f_err_eg * np.ones_like(f_obs_eg),
        observed=f_obs_eg,
    )

with model:
    soln = xo.optimize(options=dict(maxiter=99999))


# Initialize maps
map = starry.Map(ydeg_inf)
map.amp = soln["amp"]
map[1:, :] = soln["y1"]

# Normalize flux
norm = np.max(np.concatenate([soln["flux_in_dense"], soln["flux_eg_dense"]]))
f_obs_in /= norm
f_obs_eg /= norm
f_err_in /= norm
f_err_eg /= norm

flux_in_dense = soln["flux_in_dense"] / norm
flux_eg_dense = soln["flux_eg_dense"] / norm
flux_obs_in = soln["flux_pred_in"] / norm
flux_obs_eg = soln["flux_pred_eg"] / norm

# Compute residuals
res_in = f_obs_in - flux_obs_in
res_in = res_in / np.std(res_in)
res_eg = f_obs_eg - flux_obs_eg
res_eg = res_eg / np.std(res_eg)


def lon_lat_to_mollweide(lon, lat):
    lat *= np.pi / 180
    lon *= np.pi / 180

    f = lambda x: 2 * x + np.sin(2 * x) - np.pi * np.sin(lat)
    theta = optimize.newton(f, 0.3)

    x = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)

    return x, y


# Set up the plot
resol = 150
nim = 8
cmap_norm = colors.Normalize(vmin=-0.5, vmax=700)

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
ax_map_true = fig.add_subplot(gs0[0, :nim])
ax_map_inf = fig.add_subplot(gs0[0, nim:])
map_true.show(
    ax=ax_map_true, projection="molleweide", norm=cmap_norm, res=resol
)
map.show(
    ax=ax_map_inf,
    projection="molleweide",
    colorbar=True,
    norm=cmap_norm,
    res=resol,
)
ax_map_true.set_title("True map")
ax_map_inf.set_title("Inferred map")
x_spot, y_spot = lon_lat_to_mollweide((360 - 308.8), 13.0)
ax_map_inf.scatter(
    x_spot, y_spot, marker="x", color="black", s=8.0, alpha=0.7, linewidths=0.8
)

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

for j in range(2):
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
    xo_in,
    f_obs_in,
    f_err_in,
    color="black",
    fmt=".",
    ecolor="black",
    alpha=0.4,
)

ax_lc[0].plot(xo_in_dense, flux_in_dense, "C1-")  # Model

# Residuals
ax_res[0].errorbar(
    xo_in, res_in, f_err_in, color="black", fmt=".", ecolor="black", alpha=0.4,
)

# Plot egress
ax_lc[1].errorbar(
    xo_eg,
    f_obs_eg,
    f_err_eg,
    color="black",
    fmt=".",
    ecolor="black",
    alpha=0.4,
)

ax_lc[1].plot(xo_eg_dense, flux_eg_dense, "C1-")

ax_res[1].errorbar(
    xo_eg, res_eg, f_err_eg, color="black", fmt=".", ecolor="black", alpha=0.4,
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
    a.set_xticks(np.arange(-39.5, -37.0, 0.5))
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())

for a in (ax_lc[1], ax_res[1]):
    a.set_xticks(np.arange(37.0, 40.0, 0.5))
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_ticklabels([])
    a.yaxis.set_ticklabels([])
    a.set_ylabel("")
    a.set_ylabel("")

for a in ax_lc:
    a.set_ylim(-0.05, 1.05)
    a.set_xticklabels([])
    a.set_yticks(np.arange(0, 1.2, 0.2))

for a in ax_lc + ax_res:
    a.grid()

# Set common labels
fig.text(0.5, 0.04, "Occultor x position [Io radii]", ha="center", va="center")
ax_lc[0].set_ylabel("Flux")
ax_res[0].set_ylabel("Residuals\n (norm.)")

fig.savefig("ingress_egress_simulated.pdf", bbox_inches="tight", dpi=500)
