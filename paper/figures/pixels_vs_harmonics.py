import numpy as np
import os
import sys

from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
import matplotlib.cm as cm


import starry
import jax.numpy as jnp
from jax import random, jit, vmap, lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import *

from volcano.utils import get_smoothing_filter

from scipy import optimize

numpyro.enable_x64()
numpyro.set_host_device_count(4)

np.random.seed(42)
starry.config.lazy = False


def get_median_map(
    ydeg_inf,
    samples,
    projection="Mollweide",
    inc=90,
    theta=0.0,
    nsamples=100,
    resol=300,
):
    imgs = []
    map = starry.Map(ydeg=ydeg_inf)
    map.inc = inc

    for n in np.random.randint(0, len(samples), nsamples):
        x = samples[n]
        map.amp = x[0]
        map[1:, :] = x[1:] / map.amp

        if projection == "Mollweide":
            im = map.render(projection="Mollweide", res=resol)
        else:
            im = map.render(theta=theta, res=resol)
        imgs.append(im)

    return np.median(imgs, axis=0)


npts = 150
xo_eg = np.linspace(37.15, 39.43, npts)
yo_eg = np.linspace(-8.284, -8.27, npts)

xo_in = np.linspace(-39.5, -37.3, npts)
yo_in = np.linspace(-7.714, -7.726, npts)

xo_com = np.concatenate([xo_in, xo_eg])
yo_com = np.concatenate([yo_in, yo_eg])

theta_in = 350.0
theta_eg = 10.0
ro = 39.1

ydeg_true = 30
map_true = starry.Map(ydeg_true)
spot_ang_dim = 5 * np.pi / 180
spot_sigma = 1 - np.cos(spot_ang_dim / 2)
map_true.add_spot(
    amp=0.5, sigma=spot_sigma, lat=13.0, lon=51.0, relative=False
)

# Smooth the true map
sigma_s = 2 / ydeg_true
S_true = get_smoothing_filter(ydeg_true, sigma_s)
x = map_true.amp * map_true.y
xsmooth = (S_true @ x[:, None]).reshape(-1)
map_true[:, :] = xsmooth / xsmooth[0]
map_true.amp = xsmooth[0]

f_true_in = map_true.flux(ro=ro, xo=xo_in, yo=yo_in, theta=theta_in)
f_true_eg = map_true.flux(ro=ro, xo=xo_eg, yo=yo_eg, theta=theta_eg)

SN = 50
f_err = np.max(np.concatenate([f_true_in, f_true_eg])) / SN

f_err_in = f_err * np.ones_like(f_true_in)
f_err_eg = f_err * np.ones_like(f_true_eg)

f_obs_in = f_true_in + np.random.normal(0, f_err, len(f_true_in))
f_obs_eg = f_true_eg + np.random.normal(0, f_err, len(f_true_eg))
f_obs = np.concatenate([f_obs_in, f_obs_eg])


# Ylm model
ydeg_inf = 20
map = starry.Map(ydeg_inf)
lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
npix = Y2P.shape[0]

# Generate mock ingress and egress light curves
xo_in_dense = np.linspace(xo_in[0], xo_in[-1], 500)
yo_in_dense = np.linspace(yo_in[0], yo_in[-1], 500)
xo_eg_dense = np.linspace(xo_eg[0], xo_eg[-1], 500)
yo_eg_dense = np.linspace(yo_eg[0], yo_eg[-1], 500)

xo_com = np.concatenate([xo_in, xo_eg])
yo_com = np.concatenate([yo_in, yo_eg])
xo_com_dense = np.concatenate([xo_in_dense, xo_eg_dense])
yo_com_dense = np.concatenate([xo_in_dense, xo_eg_dense])

A_in = map.design_matrix(ro=ro, xo=xo_in, yo=yo_in, theta=theta_in)
A_eg = map.design_matrix(ro=ro, xo=xo_eg, yo=yo_eg, theta=theta_eg)
A_in_dense = map.design_matrix(
    ro=ro, xo=xo_in_dense, yo=yo_in_dense, theta=theta_in
)
A_eg_dense = map.design_matrix(
    ro=ro, xo=xo_eg_dense, yo=yo_eg_dense, theta=theta_eg
)


# Linear solve for Ylm posterior
map = starry.Map(ydeg=ydeg_inf)
map.set_data(f_obs, C=f_err ** 2)
mu = np.empty(map.Ny)
mu[0] = 1
mu[1:] = 0
L = np.empty(map.Ny)
L[0] = 1e0
L[1:] = 0.5 ** 2
map.set_prior(mu=mu, L=L)

# Solve
mu, cho_cov = map.solve(design_matrix=np.vstack([A_in, A_eg]))


# Sample from Ylm posterior
samples_ylm = {
    "x": [],
    "flux_in": [],
    "flux_eg": [],
    "flux_in_dense": [],
    "flux_eg_dense": [],
    "p": [],
}

for i in range(500):
    map.draw()
    x = map.amp * map._y
    samples_ylm["x"].append(x)
    samples_ylm["flux_in"].append((A_in @ x[:, None]).reshape(-1))
    samples_ylm["flux_eg"].append((A_eg @ x[:, None]).reshape(-1))
    samples_ylm["flux_in_dense"].append((A_in_dense @ x[:, None]).reshape(-1))
    samples_ylm["flux_eg_dense"].append((A_eg_dense @ x[:, None]).reshape(-1))
    samples_ylm["p"].append((Y2P @ x[:, None]).reshape(-1))


# Compute the standard deviation of pixels in the Ylm model for use as prior in pixel model
def model_ylm_prior():
    x = numpyro.sample("x", dist.Normal(mu, np.sqrt(L)))
    numpyro.deterministic("p", jnp.dot(Y2P, x[:, None]).reshape(-1))


prior_samples = Predictive(model_ylm_prior, {}, num_samples=5000)(
    random.PRNGKey(1)
)
std_p = np.std(prior_samples["p"], axis=0)


# Pixel model
def model_pix():
    p = numpyro.sample("p", dist.Exponential(1 / std_p))
    x = jnp.dot(jnp.array(P2Y), p)
    numpyro.deterministic("x", x)

    # Compute flux
    flux_in = jnp.dot(jnp.array(A_in), x[:, None]).reshape(-1)
    flux_eg = jnp.dot(jnp.array(A_eg), x[:, None]).reshape(-1)
    numpyro.deterministic("flux_in", flux_in)
    numpyro.deterministic("flux_eg", flux_eg)

    # Dense grid
    flux_in_dense = jnp.dot(jnp.array(A_in_dense), x[:, None]).reshape(-1)
    flux_eg_dense = jnp.dot(jnp.array(A_eg_dense), x[:, None]).reshape(-1)
    numpyro.deterministic("flux_in_dense", flux_in_dense)
    numpyro.deterministic("flux_eg_dense", flux_eg_dense)

    numpyro.sample(
        "obs",
        dist.Normal(
            jnp.concatenate([flux_in, flux_eg]),
            jnp.concatenate([f_err_in, f_err_eg]),
        ),
        obs=f_obs,
    )


nuts_kernel = NUTS(
    model_pix,
    dense_mass=False,
    target_accept_prob=0.95,
)

mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=1)
rng_key = random.PRNGKey(2)
mcmc.run(rng_key)
samples_pix = mcmc.get_samples()


# Compute median maps
resol = 300

median_map_moll_ylm = get_median_map(ydeg_inf, samples_ylm["x"], resol=resol)
median_map_moll_pix = get_median_map(ydeg_inf, samples_pix["x"], resol=resol)
true_map_image = map_true.render(res=resol, projection="Mollweide")

fig = plt.figure(figsize=(14, 16))

# Set up the plot
resol = 300

# True and inferred maps
gs = fig.add_gridspec(
    nrows=4,
    ncols=4,
    height_ratios=[1.5, 1.5, 2, 1],
    left=0.05,
    right=0.98,
    hspace=0.2,
    wspace=0.1,
)

ax_true_map = fig.add_subplot(gs[0, :])
ax_inf_map = [
    fig.add_subplot(gs[1, :2]),
    fig.add_subplot(gs[1, 2:]),
]

# Light curves
ax_lc = [
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
    fig.add_subplot(gs[2, 2]),
    fig.add_subplot(gs[2, 3]),
]

# Residuals
ax_res = [
    fig.add_subplot(gs[3, 0]),
    fig.add_subplot(gs[3, 1]),
    fig.add_subplot(gs[3, 2]),
    fig.add_subplot(gs[3, 3]),
]

# Plot true map
cmap_norm = colors.Normalize(vmin=0.0, vmax=50.0)
cmap = "OrRd"
map.show(
    image=true_map_image,
    ax=ax_true_map,
    projection="Mollweide",
    norm=cmap_norm,
    cmap=cmap,
)


# Plot inferred maps
map.show(
    image=median_map_moll_ylm,
    ax=ax_inf_map[0],
    projection="Mollweide",
    norm=cmap_norm,
    colorbar=False,
    cmap=cmap,
)
ax_true_map.set_title("Simulated map\n")

ax_inf_map[0].set_title("Spherical harmonic model\n(Gaussian prior)")
map.show(
    image=median_map_moll_pix,
    ax=ax_inf_map[1],
    projection="Mollweide",
    norm=cmap_norm,
    cmap=cmap,
)
ax_inf_map[1].set_title("Hybrid pixel model\n(exponential prior)")

# Plot light curve and fit
norm = np.max(np.median(samples_ylm["flux_in_dense"], axis=0))

for a in (ax_lc[0], ax_lc[2]):
    a.scatter(
        xo_in,
        f_obs_in / norm,
        color="black",
        marker="o",
        alpha=0.3,
    )

for a in (ax_lc[1], ax_lc[3]):
    a.scatter(
        xo_eg,
        f_obs_eg / norm,
        color="black",
        marker="o",
        alpha=0.3,
    )

# Plot model:
for s in np.random.randint(0, len(samples_ylm["flux_in_dense"]), 10):
    ax_lc[0].plot(
        xo_in_dense, samples_ylm["flux_in_dense"][s] / norm, "C1-", alpha=0.3
    )
    ax_lc[1].plot(
        xo_eg_dense, samples_ylm["flux_eg_dense"][s] / norm, "C1-", alpha=0.3
    )

for s in np.random.randint(0, len(samples_pix["flux_in_dense"]), 10):
    ax_lc[2].plot(
        xo_in_dense, samples_pix["flux_in_dense"][s] / norm, "C1-", alpha=0.3
    )
    ax_lc[3].plot(
        xo_eg_dense, samples_pix["flux_eg_dense"][s] / norm, "C1-", alpha=0.3
    )

# Plot residuals
res_in_ylm = (f_obs_in - np.median(samples_ylm["flux_in"], axis=0)) / norm
res_eg_ylm = (f_obs_eg - np.median(samples_ylm["flux_eg"], axis=0)) / norm

res_in_pix = (f_obs_in - np.median(samples_pix["flux_in"], axis=0)) / norm
res_eg_pix = (f_obs_eg - np.median(samples_pix["flux_eg"], axis=0)) / norm

# Ylm model
ax_res[0].errorbar(
    xo_in,
    res_in_ylm / np.std(res_in_ylm),
    (f_err_in / norm) / np.std(res_in_ylm),
    color="black",
    marker="o",
    linestyle="",
    ecolor="black",
    alpha=0.3,
)

ax_res[1].errorbar(
    xo_eg,
    res_eg_ylm / np.std(res_eg_ylm),
    (f_err_eg / norm) / np.std(res_eg_ylm),
    color="black",
    marker="o",
    linestyle="",
    ecolor="black",
    alpha=0.3,
)


# Pixel model
ax_res[2].errorbar(
    xo_in,
    res_in_pix / np.std(res_in_pix),
    (f_err_in / norm) / np.std(res_in_pix),
    color="black",
    marker="o",
    linestyle="",
    ecolor="black",
    alpha=0.3,
)

ax_res[3].errorbar(
    xo_eg,
    res_eg_pix / np.std(res_eg_pix),
    (f_err_eg / norm) / np.std(res_eg_pix),
    color="black",
    marker="o",
    linestyle="",
    ecolor="black",
    alpha=0.3,
)

for a in ax_res:
    a.xaxis.set_major_formatter(plt.FormatStrFormatter("%0.1f"))
    a.set_ylim(-4, 4)
    a.set_yticks(np.arange(-4, 6, 2))


ax_res[0].set(ylabel="Residuals\n (norm.)")

for a in ax_lc:
    a.grid(alpha=0.5)
    a.set_yticks(np.arange(0.0, 1.2, 0.2))
    a.set_ylim(-0.05, 1.05)
    a.set_xticklabels([])

for a in ax_res:
    a.grid(alpha=0.5)

ax_lc[2].set_yticklabels([])
ax_res[2].set_yticklabels([])

# Ingress ticks
for a in (ax_lc[0], ax_res[0], ax_lc[2], ax_res[2]):
    a.set_xticks(np.arange(-39.5, -37.0, 0.5))
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())
    a.set_xlim(left=-39.55, right=-37.2)

for a in (ax_lc[2], ax_res[2]):
    a.set_xticks(np.arange(-39, -37.0, 0.5))

# Egress ticks
for a in (ax_lc[1], ax_res[1], ax_lc[3], ax_res[3]):
    a.set_xticks(np.arange(37.0, 40.0, 0.5))
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_ticklabels([])
    a.yaxis.set_ticklabels([])
    a.set_xlim(left=37.05)
    a.set_ylabel("")
    a.set_ylabel("")

for a in (ax_lc[1], ax_res[1]):
    a.set_xticks(np.arange(37.0, 39.5, 0.5))

# Make broken axis
for ax in (ax_lc[:2], ax_res[:2], ax_lc[2:], ax_res[2:]):
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

ax_lc[0].set_ylabel("Intensity")

fig.text(
    0.27, 0.07, "Occultor x position [Io radii]", ha="center", va="center"
)
fig.text(
    0.75, 0.07, "Occultor x position [Io radii]", ha="center", va="center"
)

# Colorbar
cbar_ax = fig.add_axes([0.73, 0.74, 0.01, 0.11])
fig.colorbar(
    cm.ScalarMappable(norm=cmap_norm, cmap=cmap),
    cax=cbar_ax,
    ticks=[0, 25, 50],
)

# Save
fig.savefig("pixels_vs_harmonics.pdf", bbox_inches="tight", dpi=500)