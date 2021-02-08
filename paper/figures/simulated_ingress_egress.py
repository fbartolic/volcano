import numpy as np
from matplotlib import pyplot as plt

import starry
from scipy import optimize

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import *

from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator

from volcano.utils import *

np.random.seed(42)
starry.config.lazy = False
numpyro.enable_x64()

xo_eg = np.linspace(37.15, 39.43, 150)
yo_eg = np.linspace(-8.284, -8.27, 150)

xo_in = np.linspace(-39.5, -37.3, 150)
yo_in = np.linspace(-7.714, -7.726, 150)

xo_com = np.concatenate([xo_in, xo_eg])
yo_com = np.concatenate([yo_in, yo_eg])

theta_in = 350.0
theta_eg = 10.0
ro = 39.1

ydeg_true = 20
map_true = starry.Map(ydeg_true)
spot_ang_dim = 10 * np.pi / 180
spot_sigma = 1 - np.cos(spot_ang_dim / 2)
map_true.add_spot(
    amp=1.0, sigma=spot_sigma, lat=13.0, lon=51.0, relative=False
)
map_true.add_spot(
    amp=0.2, sigma=spot_sigma, lat=-15.0, lon=-40.0, relative=False
)
map_true.amp = 20

# Smooth the true map
sigma_s = 2 / ydeg_true
S_true = get_smoothing_filter(ydeg_true, sigma_s)
x = map_true.amp * map_true.y
xsmooth = (S_true @ x[:, None]).reshape(-1)
map_true[:, :] = xsmooth / xsmooth[0]
map_true.amp = xsmooth[0]

# Generate mock ingress and egress light curves
f_true_in = map_true.flux(ro=ro, xo=xo_in, yo=yo_in, theta=theta_in)
f_true_eg = map_true.flux(ro=ro, xo=xo_eg, yo=yo_eg, theta=theta_eg)

# S/N = 50
f_err_in_50 = np.max(np.concatenate([f_true_in, f_true_eg])) / 50
f_err_eg_50 = f_err_in_50
f_obs_in_50 = f_true_in + np.random.normal(0, f_err_in_50, len(f_true_in))
f_obs_eg_50 = f_true_eg + np.random.normal(0, f_err_eg_50, len(f_true_eg))

# S/N = 10
f_err_in_10 = np.max(np.concatenate([f_true_in, f_true_eg])) / 10
f_err_eg_10 = f_err_in_10
f_obs_in_10 = f_true_in + np.random.normal(0, f_err_in_10, len(f_true_in))
f_obs_eg_10 = f_true_eg + np.random.normal(0, f_err_eg_10, len(f_true_eg))

# Set up model
ydeg_inf = 20
map = starry.Map(ydeg_inf)
lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
npix = Y2P.shape[0]

Y2P = jnp.array(Y2P)
P2Y = jnp.array(P2Y)
Dx = jnp.array(Dx)
Dy = jnp.array(Dy)

# Evalute MAP model on denser grid
xo_in_dense = np.linspace(xo_in[0], xo_in[-1], 200)
yo_in_dense = np.linspace(yo_in[0], yo_in[-1], 200)
xo_eg_dense = np.linspace(xo_eg[0], xo_eg[-1], 200)
yo_eg_dense = np.linspace(yo_eg[0], yo_eg[-1], 200)

# Compute design matrix
map = starry.Map(ydeg_inf)
A_in = jnp.array(map.design_matrix(xo=xo_in, yo=yo_in, ro=ro, theta=theta_in))
A_eg = jnp.array(map.design_matrix(xo=xo_eg, yo=yo_eg, ro=ro, theta=theta_eg))
A_in_dense = jnp.array(
    map.design_matrix(xo=xo_in_dense, yo=yo_in_dense, ro=ro, theta=theta_in)
)
A_eg_dense = jnp.array(
    map.design_matrix(xo=xo_eg_dense, yo=yo_eg_dense, ro=ro, theta=theta_eg)
)


def model(f_obs_in, f_obs_eg, f_err_in, f_err_eg):
    # Set the prior scale tau0 for the global scale parameter tau
    D = npix
    N = len(f_obs_in) + len(f_obs_eg)
    peff = 0.8 * D  # Effective number of parameters
    sig = np.mean(f_err_in)
    tau0 = peff / (D - peff) * sig / np.sqrt(N)

    # Other constants for the model
    slab_scale = 10.0
    slab_df = 4

    #  Specify Finish Horseshoe prior
    # Non-centered distributions- loc=0, width=1 then shift/stretch afterwards
    beta_raw = numpyro.sample("beta_raw", dist.HalfNormal(1.0).expand([D]))
    lamda_raw = numpyro.sample("lamda_raw", dist.HalfCauchy(1.0).expand([D]))
    tau_raw = numpyro.sample("tau_raw", dist.HalfCauchy(1.0))
    c2_raw = numpyro.sample("c2_raw", dist.InverseGamma(0.5 * slab_df, 1.0))

    # Do the shifting/stretching
    tau = numpyro.deterministic("tau", tau_raw * tau0)
    c2 = numpyro.deterministic("c2", 0.5 * slab_df * slab_scale ** 2 * c2_raw)
    lamda_tilde = numpyro.deterministic(
        "lamda_tilde",
        jnp.sqrt(c2) * lamda_raw / jnp.sqrt(c2 + tau ** 2 * lamda_raw ** 2),
    )
    numpyro.deterministic(
        "mu_meff", (tau / sig * np.sqrt(N)) / (1 + tau / sig * np.sqrt(N)) * D
    )

    # The Finnish Horseshoe prior on p
    p = numpyro.deterministic("p", tau * lamda_tilde * beta_raw)
    x = jnp.dot(P2Y, p)

    # Run the smoothing filter
    S = jnp.array(get_smoothing_filter(ydeg_inf, 2 / ydeg_inf))
    x_s = jnp.dot(S, x[:, None]).reshape(-1)
    numpyro.deterministic("x", x_s)

    # Compute flux
    flux_in = jnp.dot(A_in, x_s[:, None]).reshape(-1)
    flux_eg = jnp.dot(A_eg, x_s[:, None]).reshape(-1)
    numpyro.deterministic("flux_pred_in", flux_in)
    numpyro.deterministic("flux_pred_eg", flux_eg)
    flux = jnp.concatenate([flux_in, flux_eg])

    # Dense grid
    flux_in_dense = jnp.dot(A_in_dense, x_s[:, None]).reshape(-1)
    flux_eg_dense = jnp.dot(A_eg_dense, x_s[:, None]).reshape(-1)
    numpyro.deterministic("flux_in_dense", flux_in_dense)
    numpyro.deterministic("flux_eg_dense", flux_eg_dense)
    flux_dense = jnp.concatenate([flux_in_dense, flux_eg_dense])

    numpyro.sample(
        "obs_in",
        dist.Normal(flux_in, f_err_in * np.ones_like(f_obs_in),),
        obs=f_obs_in,
    )

    numpyro.sample(
        "obs_eg",
        dist.Normal(flux_eg, f_err_eg * np.ones_like(f_obs_eg),),
        obs=f_obs_eg,
    )


init_vals = {
    "beta_raw": 0.5 * jnp.ones(npix),
    "lamda": jnp.ones(npix),
    "tau_raw": 0.1,
    "c2_raw": 5 ** 2,
}

nuts_kernel = NUTS(
    model,
    dense_mass=False,
    init_strategy=init_to_value(values=init_vals),
    target_accept_prob=0.95,
)

# Run MCMC
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1500)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, f_obs_in_50, f_obs_eg_50, f_err_in_50, f_err_eg_50)
samples_50 = mcmc.get_samples()

mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1500)
rng_key = random.PRNGKey(1)
mcmc.run(rng_key, f_obs_in_10, f_obs_eg_10, f_err_in_10, f_err_eg_10)
samples_10 = mcmc.get_samples()

# Compute median maps
median_map_moll_50 = get_median_map(ydeg_inf, samples_50["x"])
median_map_in_50 = get_median_map(
    ydeg_inf, samples_50["x"], projection=None, theta=theta_in, nsamples=15
)
median_map_eg_50 = get_median_map(
    ydeg_inf, samples_50["x"], projection=None, theta=theta_eg, nsamples=15
)

median_map_moll_10 = get_median_map(ydeg_inf, samples_10["x"])
median_map_in_10 = get_median_map(
    ydeg_inf, samples_10["x"], projection=None, theta=theta_in, nsamples=15
)
median_map_eg_10 = get_median_map(
    ydeg_inf, samples_10["x"], projection=None, theta=theta_eg, nsamples=15
)


def lon_lat_to_mollweide(lon, lat):
    lat *= np.pi / 180
    lon *= np.pi / 180

    f = lambda x: 2 * x + np.sin(2 * x) - np.pi * np.sin(lat)
    theta = optimize.newton(f, 0.3)

    x = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)

    return x, y


def plot(
    f_obs_in,
    f_obs_eg,
    f_err_in,
    f_err_eg,
    samples,
    median_map_moll,
    median_map_in,
    median_map_eg,
    ylim_low,
    ylim_high,
    fname,
):
    # Normalize flux
    norm = np.max(
        np.concatenate(
            [
                np.median(samples["flux_in_dense"], axis=0),
                np.median(samples["flux_eg_dense"], axis=0),
            ]
        )
    )
    f_obs_in /= norm
    f_obs_eg /= norm
    f_err_in /= norm
    f_err_eg /= norm

    flux_obs_in = np.median(samples["flux_pred_in"], axis=0) / norm
    flux_obs_eg = np.median(samples["flux_pred_eg"], axis=0) / norm

    # Compute residuals
    res_in = f_obs_in - flux_obs_in
    res_eg = f_obs_eg - flux_obs_eg

    fig = plt.figure(figsize=(8, 10))

    # Set up the plot
    nim = 7
    cmap_norm = colors.Normalize(vmin=-0.5, vmax=300)
    cmap = "Oranges"
    resol = 300

    # True and inferred maps
    gs1 = fig.add_gridspec(
        nrows=2,
        ncols=2 * nim,
        height_ratios=[1, 1],
        bottom=0.52,
        left=0.05,
        right=0.98,
        hspace=0.3,
    )
    # Minimaps, light curves and residuals ingress
    gs2 = fig.add_gridspec(
        nrows=3,
        ncols=nim,
        height_ratios=[1, 3, 1],
        top=0.5,
        left=0.05,
        right=0.50,
        hspace=0.05,
    )
    # Minimaps, light curves and residuals ingress
    gs3 = fig.add_gridspec(
        nrows=3,
        ncols=nim,
        height_ratios=[1, 3, 1],
        top=0.5,
        left=0.53,
        right=0.98,
        hspace=0.05,
    )

    # Inferred map
    ax_true_map = fig.add_subplot(gs1[0, :])
    ax_inf_map = fig.add_subplot(gs1[1, :])

    # Minimaps
    ax_im = [
        [fig.add_subplot(gs2[0, i]) for i in range(nim)],
        [fig.add_subplot(gs3[0, i]) for i in range(nim)],
    ]
    # Light curves
    ax_lc = [
        fig.add_subplot(gs2[1, :]),
        fig.add_subplot(gs3[1, :]),
    ]
    # Residuals
    ax_res = [
        fig.add_subplot(gs2[2, :]),
        fig.add_subplot(gs3[2, :]),
    ]

    #  True map
    map_true.show(
        ax=ax_true_map,
        projection="molleweide",
        norm=cmap_norm,
        colorbar=True,
        res=resol,
        cmap=cmap,
    )
    ax_true_map.set_title("True map")

    # Inferred mean map
    im = map.show(
        image=median_map_moll,
        ax=ax_inf_map,
        projection="Mollweide",
        norm=cmap_norm,
        colorbar=True,
        cmap=cmap,
    )
    ax_inf_map.set_title("Inferred map")

    x_spot, y_spot = lon_lat_to_mollweide(51.0, 13.0)
    ax_inf_map.scatter(
        x_spot, y_spot, marker="x", color="black", s=10.0, alpha=0.3,
    )
    x_spot2, y_spot2 = lon_lat_to_mollweide(-40.0, -15.0)
    ax_inf_map.scatter(
        x_spot2, y_spot2, marker="x", color="black", s=10.0, alpha=0.3,
    )

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
            if j == 0:
                map.show(
                    image=median_map_in,
                    ax=a[n],
                    grid=False,
                    norm=cmap_norm,
                    cmap=cmap,
                )
            else:
                map.show(
                    image=median_map_eg,
                    ax=a[n],
                    grid=False,
                    norm=cmap_norm,
                    cmap=cmap,
                )

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
        marker=".",
        linestyle="",
        ecolor="black",
        alpha=0.4,
    )

    for s in np.random.randint(0, len(samples["flux_in_dense"]), 10):
        ax_lc[0].plot(
            xo_in_dense,
            samples["flux_in_dense"][s, :] / norm,
            "C1-",
            alpha=0.2,
        )  # Model

    # Residuals
    ax_res[0].errorbar(
        xo_in,
        res_in / np.std(res_in),
        f_err_in / np.std(res_in),
        color="black",
        marker=".",
        linestyle="",
        ecolor="black",
        alpha=0.4,
    )

    # Plot egress
    ax_lc[1].errorbar(
        xo_eg,
        f_obs_eg,
        f_err_eg,
        color="black",
        marker=".",
        linestyle="",
        ecolor="black",
        alpha=0.4,
    )

    for s in np.random.randint(0, len(samples["flux_eg_dense"]), 10):
        ax_lc[1].plot(
            xo_eg_dense,
            samples["flux_eg_dense"][s, :] / norm,
            "C1-",
            alpha=0.2,
        )  # Model

    ax_res[1].errorbar(
        xo_eg,
        res_eg / np.std(res_eg),
        f_err_eg / np.std(res_eg),
        color="black",
        marker=".",
        linestyle="",
        ecolor="black",
        alpha=0.4,
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
        a.set_xlim(left=-39.55, right=-37.2)

    for a in (ax_lc[1], ax_res[1]):
        a.set_xticks(np.arange(37.0, 40.0, 0.5))
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_ticklabels([])
        a.yaxis.set_ticklabels([])
        a.set_xlim(left=37.05)
        a.set_ylabel("")
        a.set_ylabel("")

    for a in ax_lc:
        a.set_ylim(ylim_low, ylim_high)
        a.set_xticklabels([])
        a.set_yticks(np.arange(0, 1.2, 0.2))

    for a in ax_lc + ax_res:
        a.grid()

    # Set common labels
    fig.text(
        0.5, 0.06, "Occultor x position [Io radii]", ha="center", va="center"
    )
    ax_lc[0].set_ylabel("Flux")
    ax_res[0].set_ylabel("Residuals\n (norm.)")

    fig.savefig(fname, bbox_inches="tight", dpi=500)


plot(
    f_obs_in_50,
    f_obs_eg_50,
    f_err_in_50,
    f_err_eg_50,
    samples_50,
    median_map_moll_50,
    median_map_in_50,
    median_map_eg_50,
    -0.1,
    1.1,
    "ingress_egress_sim_snr_50.pdf",
)

plot(
    f_obs_in_10,
    f_obs_eg_10,
    f_err_in_10,
    f_err_eg_10,
    samples_10,
    median_map_moll_10,
    median_map_in_10,
    median_map_eg_10,
    -0.25,
    1.25,
    "ingress_egress_sim_snr_10.pdf",
)
