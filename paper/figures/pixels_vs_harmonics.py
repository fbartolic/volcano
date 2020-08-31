import numpy as np
from matplotlib import pyplot as plt

import theano
import theano.tensor as tt
import pymc3 as pm
import starry
import exoplanet as xo

from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator

from volcano.utils import *

np.random.seed(42)
starry.config.lazy = True

# Set up mock map
def get_S(ydeg, sigma=0.1):
    l = np.concatenate([np.repeat(l, 2 * l + 1) for l in range(ydeg + 1)])
    s = np.exp(-0.5 * l * (l + 1) * sigma ** 2)
    S = np.diag(s)
    return S


xo_sim = np.linspace(37.15, 39.43, 120)
yo_sim = np.linspace(-8.284, -8.27, 120)
ro = 39.1

ydeg_true = 30
map_true = starry.Map(ydeg_true)
map_true.add_spot(
    amp=2.0, sigma=1e-03, lat=13.0, lon=360 - 308.8, relative=False
)
map_true.amp = 20

# Smooth the true map
S_true = get_S(ydeg_true, 0.07)
x = (map_true.amp * map_true.y).eval()
x_smooth = (S_true @ x[:, None]).reshape(-1)
map_true[:, :] = x_smooth / x_smooth[0]
map_true.amp = x_smooth[0]

# Generate mock light curve
f_true = map_true.flux(ro=ro, xo=xo_sim, yo=yo_sim).eval()
f_err = 0.5
f_obs = f_true + np.random.normal(0, f_err, len(f_true))

# Ylm model
ydeg_inf = 20
map = starry.Map(ydeg_inf)
lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
npix = Y2P.shape[0]

# Evalute MAP model on denser grid
xo_dense = np.linspace(xo_sim[0], xo_sim[-1], 200)
yo_dense = np.linspace(yo_sim[0], yo_sim[-1], 200)

PositiveNormal = pm.Bound(pm.Normal, lower=0.0)
ncoeff = (ydeg_inf + 1) ** 2

with pm.Model() as model_ylm:
    map = starry.Map(ydeg_inf)

    cov = (3e-02) ** 2 * np.eye(ncoeff - 1)
    y1 = pm.MvNormal(
        "y1",
        np.zeros(ncoeff - 1),
        cov,
        shape=(ncoeff - 1,),
        testval=1e-03 * np.random.rand(ncoeff - 1),
    )  # testval=['y1'])
    ln_amp = pm.Normal("ln_amp", 0.0, 5.0, testval=np.log(20.0))
    pm.Deterministic("amp", tt.exp(ln_amp))

    map.amp = tt.exp(ln_amp)
    map[1:, :] = y1

    flux = map.flux(xo=theano.shared(xo_sim), yo=theano.shared(yo_sim), ro=ro)
    pm.Deterministic("flux_pred", flux)

    # Dense grid
    MAP_flux_ylm = map.flux(
        xo=theano.shared(xo_dense), yo=theano.shared(yo_dense), ro=ro
    )
    pm.Deterministic("flux_dense", flux)

    pm.Normal("obs", mu=flux, sd=f_err * tt.ones(len(f_obs)), observed=f_obs)

with model_ylm:
    soln_ylm = xo.optimize(options=dict(maxiter=99999))

# Pixel model
with pm.Model() as model_pix:
    map = starry.Map(ydeg_inf)

    p = pm.Exponential("p", 1 / 10.0, shape=(npix,))
    x = tt.dot(P2Y, p)

    map.amp = x[0]
    map[1:, :] = x[1:] / map.amp

    pm.Deterministic("amp", map.amp)
    pm.Deterministic("y1", map[1:, :])

    # Compute flux
    flux = map.flux(xo=theano.shared(xo_sim), yo=theano.shared(yo_sim), ro=ro)
    pm.Deterministic("flux_pred", flux)

    # Dense grid
    MAP_flux_pix = map.flux(
        xo=theano.shared(xo_dense), yo=theano.shared(yo_dense), ro=ro
    )
    pm.Deterministic("flux_dense", flux)

    pm.Normal("obs", mu=flux, sd=f_err * np.ones_like(f_obs), observed=f_obs)

with model_pix:
    soln_pix = xo.optimize(options=dict(maxiter=99999))

# Compute value of pixels at MAP estimate
lat_true, lon_true, Y2P_true, P2Y_true, _, _ = map_true.get_pixel_transforms(
    oversample=4
)

p_true = np.dot(
    Y2P_true, (map_true.amp.eval() * map_true.y.eval())[:, None]
).reshape(-1)
x_inf_ylm = soln_ylm["amp"] * soln_ylm["y1"]
x_inf_ylm = np.insert(x_inf_ylm, 0, soln_ylm["amp"], axis=0)
p_inf_ylm = np.dot(Y2P, x_inf_ylm[:, None]).reshape(-1)

x_inf = soln_pix["amp"] * soln_pix["y1"]
x_inf = np.insert(x_inf, 0, soln_pix["amp"], axis=0)
p_inf = np.dot(Y2P, x_inf[:, None]).reshape(-1)

fig, ax = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
fig.subplots_adjust(wspace=0.15)

bins = "fd"
# Ylm model
ax[0].hist(
    p_true,
    density=True,
    cumulative=True,
    bins=bins,
    histtype="step",
    color="C0",
    lw=2.0,
    label="True",
    alpha=0.9,
)
ax[0].hist(
    p_inf_ylm,
    density=True,
    cumulative=True,
    bins=bins,
    histtype="step",
    color="C1",
    lw=2.0,
    label="Inferred",
    alpha=0.9,
)

# Pixel model
ax[1].hist(
    p_true,
    density=True,
    cumulative=True,
    bins=bins,
    alpha=0.9,
    histtype="step",
    color="C0",
    lw=2.0,
    label="True",
)
ax[1].hist(
    soln_pix["p"],
    density=True,
    cumulative=True,
    bins=bins,
    alpha=0.9,
    histtype="step",
    color="C2",
    lw=2.0,
    label="Schmixels",
)
ax[1].hist(
    p_inf,
    density=True,
    cumulative=True,
    bins=bins,
    alpha=0.9,
    histtype="step",
    color="C1",
    lw=2.0,
    label="Inferred",
)

for a in ax.flatten():
    a.set_xlim(-50, 125)
    a.axvline(0.0, color="black", linestyle="--", alpha=0.5)
    a.set_xticks(np.arange(-50, 150, 25))


fig.text(-0.07, 0.5, "Cummulative density", va="center", rotation="vertical")

ax[0].set_title("$Y_{lm}$ model")
ax[1].set_title("Pixel model")
ax[1].set_xlabel("Pixel value")

ax[0].legend(loc=4, prop={"size": 12})
ax[1].legend(loc=4, prop={"size": 12})

# Save
fig.savefig("pixels_hist.pdf", bbox_inches="tight", dpi=500)

def plot_everything(
    map, ax_map, ax_im, ax_lc, ax_res, soln, flux_dense, residuals
):
    t_dense = np.linspace(xo_im[0], xo_im[-1], len(soln["flux_dense"]))
    nim = len(ax_im)
    resol = 300

    # Plot map
    map.show(
        projection="molleweide",
        ax=ax_map,
        colorbar=True,
        norm=colors.Normalize(vmin=-0.5),
    )
    ax_map.axis("off")

    # Plot mini maps
    for n in range(nim):
        # Show the image
        map.show(
            ax=ax_im[n], res=resol, grid=False,
        )

        # Outline
        x = np.linspace(-1, 1, 1000)
        y = np.sqrt(1 - x ** 2)
        f = 0.98
        ax_im[n].plot(f * x, f * y, "k-", lw=0.5, zorder=0)
        ax_im[n].plot(f * x, -f * y, "k-", lw=0.5, zorder=0)

        # Occultor
        x = np.linspace(-1.5, xo_im[n] + ro - 1e-5, resol)
        y = np.sqrt(ro ** 2 - (x - xo_im[n]) ** 2)
        ax_im[n].fill_between(
            x,
            yo_im[n] - y,
            yo_im[n] + y,
            fc="w",
            zorder=1,
            clip_on=True,
            ec="k",
            lw=0.5,
        )
        ax_im[n].axis("off")
        ax_im[n].set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        ax_im[n].set_rasterization_zorder(0)

    # Plot light curve and fit
    ax_lc.errorbar(
        xo_sim,
        f_obs,
        f_err,
        color="black",
        fmt=".",
        ecolor="black",
        alpha=0.4,
    )
    ax_lc.plot(t_dense, flux_dense, "C1-")
    ax_lc.set_ylim(bottom=0.0)
    ax_lc.set_ylabel("Flux")
    ax_lc.set_xticklabels([])

    # Residuals
    ax_res.errorbar(
        xo_sim,
        residuals,
        f_err,
        color="black",
        fmt=".",
        ecolor="black",
        alpha=0.4,
    )
    ax_res.set(ylabel="Residuals\n (norm.)")
    ax_res.xaxis.set_major_formatter(plt.FormatStrFormatter("%0.1f"))

    # Appearance
    ax_im[-1].set_zorder(-100)


# Normalize flux
norm = np.max(soln_ylm["flux_dense"])
f_obs /= norm
f_err /= norm
MAP_flux_ylm /= norm
MAP_flux_pix /= norm

MAP_obs_ylm = soln_ylm["flux_pred"] / norm
MAP_obs_pix = soln_pix["flux_pred"] / norm

flux_dense_ylm = soln_ylm["flux_dense"] / norm
flux_dense_pix = soln_pix["flux_dense"] / norm

# Compute residuals
res_ylm = f_obs - MAP_obs_ylm
res_ylm = res_ylm / np.std(res_ylm)

res_pix = f_obs - MAP_obs_pix
res_pix = res_pix / np.std(res_pix)


# Set up the plot
nim = 8

# Occultation params for mini subplots
xo_im = np.linspace(xo_sim[0], xo_sim[-1], nim)
yo_im = np.linspace(yo_sim[0], yo_sim[-1], nim)

# Initialize maps
map_ylm = starry.Map(ydeg_inf)
map_ylm.amp = soln_ylm["amp"]
map_ylm[1:, :] = soln_ylm["y1"]

map_pix = starry.Map(ydeg_inf)
map_pix.amp = soln_pix["amp"]
map_pix[1:, :] = soln_pix["y1"]

fig = plt.figure(figsize=(8, 10))
fig.subplots_adjust(wspace=0.0)

heights = [3, 2, 3, 1]
gs0 = fig.add_gridspec(
    nrows=1, ncols=2 * nim, bottom=0.735, left=0.05, right=0.98,
)
gs1 = fig.add_gridspec(
    nrows=4, ncols=nim, height_ratios=heights, top=0.6, left=0.05, right=0.48
)
gs2 = fig.add_gridspec(
    nrows=4, ncols=nim, height_ratios=heights, top=0.6, left=0.55, right=0.98
)

# True map subplot
ax_true_map = fig.add_subplot(gs0[0, :])
map_true.show(ax=ax_true_map, projection="molleweide", colorbar=True)
ax_true_map.axis("off")
ax_true_map.set_title("True map")

# Inferred maps
ax_map = [fig.add_subplot(gs1[0, :]), fig.add_subplot(gs2[0, :])]

# Minimaps
ax_im = [
    [fig.add_subplot(gs1[1, i]) for i in range(nim)],
    [fig.add_subplot(gs2[1, i]) for i in range(nim)],
]

# Light curves
ax_lc = [fig.add_subplot(gs1[2, :]), fig.add_subplot(gs2[2, :])]

# Residuals
ax_res = [fig.add_subplot(gs1[3, :]), fig.add_subplot(gs2[3, :])]

plot_everything(
    map_ylm,
    ax_map[0],
    ax_im[0],
    ax_lc[0],
    ax_res[0],
    soln_ylm,
    flux_dense_ylm,
    res_ylm,
)
plot_everything(
    map_pix,
    ax_map[1],
    ax_im[1],
    ax_lc[1],
    ax_res[1],
    soln_pix,
    flux_dense_pix,
    res_pix,
)

ax_lc[1].set_ylabel("")
ax_res[1].set_ylabel("")
ax_lc[1].yaxis.set_ticklabels([])
ax_res[1].yaxis.set_ticklabels([])

fig.text(0.5, 0.05, "Occultor x-position [Io radii]", ha="center")

ax_map[1].set_title("Exponential prior on\n pixels", pad=20)
ax_map[0].set_title(
    "Gaussian prior on $\mathrm{Y}_{lm}$\n coefficients", pad=20
)

# Ticks
for a in ax_lc:
    a.set_ylim(-0.05, 1.05)
    a.set_xticklabels([])
    a.set_yticks(np.arange(0, 1.2, 0.2))

for a in (ax_lc + ax_res):
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())
    a.set_xticks(np.arange(37., 40., 0.5))
    a.grid()

for a in ax_res:
    a.set_ylim(-5., 5.)

# Save
fig.savefig("pixels_vs_harmonics.pdf", bbox_inches="tight", dpi=500)
