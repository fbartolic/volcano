import numpy as np
import os
import sys

from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator

import starry
import jax.numpy as jnp
from jax import random, jit, vmap, lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import *

from volcano.utils import get_smoothing_filter

numpyro.enable_x64()

np.random.seed(42)
starry.config.lazy = True


def get_median_map(
    ydeg_inf,
    samples,
    projection="Mollweide",
    inc=90,
    theta=0.0,
    nsamples=20,
    resol=300,
):
    imgs = []
    map = starry.Map(ydeg=ydeg_inf)
    map.inc = inc

    for n in np.random.randint(0, len(samples), 10):
        x = samples[n]
        map.amp = x[0]
        map[1:, :] = x[1:] / map.amp

        if projection == "Mollweide":
            im = map.render(projection="Mollweide", res=resol).eval()
        else:
            im = map.render(theta=theta, res=resol).eval()
        imgs.append(im)

    return np.median(imgs, axis=0)


# Set up mock map
xo_sim = np.linspace(37.15, 39.43, 120)
yo_sim = np.linspace(-8.284, -8.27, 120)
ro = 39.1

ydeg_true = 30
map_true = starry.Map(ydeg_true)
spot_ang_dim = 5 * np.pi / 180
spot_sigma = 1 - np.cos(spot_ang_dim / 2)
map_true.add_spot(
    amp=1.0, sigma=spot_sigma, lat=13.0, lon=360 - 309, relative=False
)

# Smooth the true map
sigma_s = 2 / ydeg_true
S_true = get_smoothing_filter(ydeg_true, sigma_s)
x = (map_true.amp * map_true.y).eval()
x_smooth = (S_true @ x[:, None]).reshape(-1)
map_true[:, :] = x_smooth / x_smooth[0]
map_true.amp = x_smooth[0]

# Generate mock light curve
f_true = map_true.flux(ro=ro, xo=xo_sim, yo=yo_sim).eval()
SN = 50
f_err = np.max(f_true) / SN
f_obs = f_true + np.random.normal(0, f_err, len(f_true))


# Ylm model
ydeg_inf = 20
map = starry.Map(ydeg_inf)
lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
npix = Y2P.shape[0]

Y2P = jnp.array(Y2P)
P2Y = jnp.array(P2Y)
Dx = jnp.array(Dx)
Dy = jnp.array(Dy)

# Evalute MAP model on denser grid
xo_dense = np.linspace(xo_sim[0], xo_sim[-1], 200)
yo_dense = np.linspace(yo_sim[0], yo_sim[-1], 200)

ncoeff = (ydeg_inf + 1) ** 2

# Compute design matrix
map = starry.Map(ydeg_inf)
A = jnp.array(map.design_matrix(xo=xo_sim, yo=yo_sim, ro=ro).eval())
A_dense = jnp.array(map.design_matrix(xo=xo_dense, yo=yo_dense, ro=ro).eval())


def model_ylm():
    y1 = numpyro.sample(
        "y1",
        dist.Normal(jnp.zeros(ncoeff - 1), 1e-01 * jnp.ones(ncoeff - 1)),
    )
    amp = numpyro.sample("amp", dist.LogNormal(0.0, 1.0))

    x = amp * jnp.concatenate([jnp.array([1.0]), y1], axis=0)
    numpyro.deterministic("x", x)

    flux = jnp.dot(A, x[:, None]).reshape(-1)
    numpyro.deterministic("flux_pred", flux)

    # Dense grid
    flux_dense = jnp.dot(A_dense, x[:, None]).reshape(-1)
    numpyro.deterministic("flux_dense", flux)

    numpyro.sample(
        "obs", dist.Normal(flux, f_err * jnp.ones(len(f_obs))), obs=f_obs
    )


init_vals = {"y1": 1e-03 * np.random.rand(ncoeff - 1), "amp": 4.0}

nuts_kernel = NUTS(
    model_ylm,
    dense_mass=False,
    init_strategy=init_to_value(values=init_vals),
    target_accept_prob=0.95,
)

mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key)
samples_ylm = mcmc.get_samples()

# Compute the standard deviation of pixels in the Ylm model
def model_ylm_prior():
    y1 = numpyro.sample(
        "y1",
        dist.Normal(jnp.zeros(ncoeff - 1), 0.5 * 1e-01 * jnp.ones(ncoeff - 1)),
    )
    amp = numpyro.sample("amp", dist.LogNormal(0.0, 1.0))
    x = numpyro.deterministic(
        "x", amp * jnp.concatenate([jnp.array([1.0]), y1])
    )
    numpyro.deterministic("pix", jnp.dot(Y2P, x[:, None]).reshape(-1))


prior_samples = Predictive(model_ylm_prior, {}, num_samples=10000)(
    random.PRNGKey(1)
)
std_p = np.std(np.concatenate(prior_samples["pix"]))

# Pixel model gaussian prior
def model_sphix_gauss():
    p = numpyro.sample("p", dist.HalfNormal(std_p).expand([npix]))
    x = jnp.dot(P2Y, p)
    numpyro.deterministic("x", x)

    # Compute flux
    flux = jnp.dot(A, x[:, None]).reshape(-1)
    numpyro.deterministic("flux_pred", flux)

    # Dense grid
    flux_dense = jnp.dot(A_dense, x[:, None]).reshape(-1)
    numpyro.deterministic("flux_dense", flux)

    numpyro.sample(
        "obs", dist.Normal(flux, f_err * jnp.ones(len(f_obs))), obs=f_obs
    )


nuts_kernel = NUTS(
    model_sphix_gauss,
    dense_mass=False,
    target_accept_prob=0.95,
)

mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000)
rng_key = random.PRNGKey(2)
mcmc.run(rng_key)
samples_sphix_gauss = mcmc.get_samples()

# Pixel model exponential prior
def model_sphix_exp():
    p = numpyro.sample("p", dist.Exponential(1 / std_p).expand([npix]))
    x = jnp.dot(P2Y, p)
    numpyro.deterministic("x", x)

    # Compute flux
    flux = jnp.dot(A, x[:, None]).reshape(-1)
    numpyro.deterministic("flux_pred", flux)

    # Dense grid
    flux_dense = jnp.dot(A_dense, x[:, None]).reshape(-1)
    numpyro.deterministic("flux_dense", flux)

    numpyro.sample(
        "obs", dist.Normal(flux, f_err * jnp.ones(len(f_obs))), obs=f_obs
    )


nuts_kernel = NUTS(
    model_sphix_exp,
    dense_mass=False,
    target_accept_prob=0.95,
)

mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000)
rng_key = random.PRNGKey(3)
mcmc.run(rng_key)
samples_sphix_exp = mcmc.get_samples()

# Compute median maps
median_map_moll_ylm = get_median_map(ydeg_inf, samples_ylm["x"], nsamples=30)
median_map_ylm = get_median_map(
    ydeg_inf, samples_ylm["x"], projection=None, nsamples=30
)

median_map_moll_sphix_gauss = get_median_map(
    ydeg_inf, samples_sphix_gauss["x"], nsamples=30
)
median_map_sphix_gauss = get_median_map(
    ydeg_inf, samples_sphix_gauss["x"], projection=None, nsamples=30
)

median_map_moll_sphix_exp = get_median_map(
    ydeg_inf, samples_sphix_exp["x"], nsamples=30
)
median_map_sphix_exp = get_median_map(
    ydeg_inf, samples_sphix_exp["x"], projection=None, nsamples=30
)

norm = np.max(np.median(samples_ylm["flux_dense"], axis=0))
cmap = "Oranges"


def plot_everything(
    median_map_moll,
    median_map,
    ax_map,
    ax_im,
    ax_lc,
    ax_res,
    samples,
    residuals,
    show_cbar=True,
    cmap_norm=colors.Normalize(vmin=-0.5),
):
    t_dense = np.linspace(xo_im[0], xo_im[-1], len(samples["flux_dense"][-1]))
    nim = len(ax_im)

    # Plot map
    im = map.show(
        image=median_map_moll,
        ax=ax_map,
        projection="Mollweide",
        norm=cmap_norm,
        colorbar=show_cbar,
        cmap=cmap,
    )
    ax_map.axis("off")

    # Plot mini maps
    for n in range(nim):
        # Show the image
        map.show(
            image=median_map,
            ax=ax_im[n],
            grid=False,
            norm=cmap_norm,
            cmap=cmap,
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
        f_obs / norm,
        f_err / norm,
        color="black",
        marker=".",
        linestyle="",
        ecolor="black",
        alpha=0.4,
    )

    for s in np.random.randint(0, len(samples["flux_dense"]), 10):
        ax_lc.plot(
            t_dense, samples["flux_dense"][s, :] / norm, "C1-", alpha=0.3
        )  # Model

    ax_lc.set_ylim(bottom=0.0)
    ax_lc.set_ylabel("Flux")
    ax_lc.set_xticklabels([])

    # Residuals
    ax_res.errorbar(
        xo_sim,
        residuals,
        (f_err / norm),
        color="black",
        marker=".",
        linestyle="",
        ecolor="black",
        alpha=0.4,
    )
    ax_res.set(ylabel="Residuals")
    ax_res.xaxis.set_major_formatter(plt.FormatStrFormatter("%0.1f"))
    #    ax_res.set_ylim(-3, 3)
    #    ax_res.set_yticks([-3, 0, 3])

    # Appearance
    ax_im[-1].set_zorder(-100)


# Compute  residuals
res_ylm = (f_obs - np.median(samples_ylm["flux_dense"], axis=0)) / norm
res_pix_gauss = (
    f_obs - np.median(samples_sphix_gauss["flux_dense"], axis=0)
) / norm
res_pix_exp = (
    f_obs - np.median(samples_sphix_exp["flux_dense"], axis=0)
) / norm

# Set up the plot
nim = 8
resol = 300

# Occultation params for mini subplots
xo_im = np.linspace(xo_sim[0], xo_sim[-1], nim)
yo_im = np.linspace(yo_sim[0], yo_sim[-1], nim)

fig = plt.figure(figsize=(14, 14))

heights = [3, 2, 4, 2]
gs0 = fig.add_gridspec(
    nrows=1,
    ncols=3 * nim,
    bottom=0.74,
    left=0.05,
    right=0.98,
    hspace=0.0,
    wspace=0.1,
)
gs1 = fig.add_gridspec(
    nrows=4,
    ncols=nim,
    height_ratios=heights,
    top=0.68,
    left=0.03,
    right=0.33,
    hspace=0.1,
    wspace=0.1,
)
gs2 = fig.add_gridspec(
    nrows=4,
    ncols=nim,
    height_ratios=heights,
    top=0.68,
    left=0.35,
    right=0.66,
    hspace=0.1,
    wspace=0.1,
)
gs3 = fig.add_gridspec(
    nrows=4,
    ncols=nim,
    height_ratios=heights,
    top=0.68,
    left=0.68,
    right=0.98,
    hspace=0.1,
    wspace=0.1,
)

# True map subplot
ax_true_map = fig.add_subplot(gs0[0, :])
map_true.show(
    ax=ax_true_map,
    projection="molleweide",
    colorbar=True,
    res=resol,
    cmap=cmap,
    #    norm=colors.Normalize(vmin=-0.5),
)
ax_true_map.axis("off")
ax_true_map.set_title("True map")

# Inferred maps
ax_map = [
    fig.add_subplot(gs1[0, :]),
    fig.add_subplot(gs2[0, :]),
    fig.add_subplot(gs3[0, :]),
]

# Minimaps
ax_im = [
    [fig.add_subplot(gs1[1, i]) for i in range(nim)],
    [fig.add_subplot(gs2[1, i]) for i in range(nim)],
    [fig.add_subplot(gs3[1, i]) for i in range(nim)],
]

# Light curves
ax_lc = [
    fig.add_subplot(gs1[2, :]),
    fig.add_subplot(gs2[2, :]),
    fig.add_subplot(gs3[2, :]),
]

# Residuals
ax_res = [
    fig.add_subplot(gs1[3, :]),
    fig.add_subplot(gs2[3, :]),
    fig.add_subplot(gs3[3, :]),
]

plot_everything(
    median_map_moll_ylm,
    median_map_ylm,
    ax_map[0],
    ax_im[0],
    ax_lc[0],
    ax_res[0],
    samples_ylm,
    res_ylm,
    show_cbar=False,
    cmap_norm=colors.Normalize(vmin=-0.5, vmax=20),
)
plot_everything(
    median_map_moll_sphix_gauss,
    median_map_sphix_gauss,
    ax_map[1],
    ax_im[1],
    ax_lc[1],
    ax_res[1],
    samples_sphix_gauss,
    res_pix_gauss,
    show_cbar=False,
    cmap_norm=colors.Normalize(vmin=-0.5, vmax=20),
)
plot_everything(
    median_map_moll_sphix_exp,
    median_map_sphix_exp,
    ax_map[2],
    ax_im[2],
    ax_lc[2],
    ax_res[2],
    samples_sphix_exp,
    res_pix_exp,
    show_cbar=True,
    cmap_norm=colors.Normalize(vmin=-0.5, vmax=20),
)

for a in ax_lc[1:] + ax_res[1:]:
    a.yaxis.set_ticklabels([])
    a.yaxis.set_ticklabels([])
    a.set_ylabel("")
    a.set_ylabel("")

fig.text(0.5, 0.06, "Occultor x-position [Io radii]", ha="center")

ax_map[0].set_title(
    "Gaussian prior on $\mathrm{Y}_{lm}$\n coefficients", pad=20
)
ax_map[1].set_title("Positive Gaussian\n prior on sphixels", pad=20)
ax_map[2].set_title("Exponential\n prior on sphixels", pad=20)

#  Ticks
for a in ax_lc:
    a.set_ylim(-0.1, 1.1)
    a.set_xticklabels([])
    a.set_yticks(np.arange(0, 1.2, 0.2))

for a in ax_lc + ax_res:
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())
    #    a.set_xticks(np.arange(37.0, 40.0, 0.5))
    a.grid()

# Save
fig.savefig("pixels_vs_harmonics.pdf", bbox_inches="tight", dpi=500)