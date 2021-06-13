import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl

import starry
import celerite2.jax
from celerite2.jax import terms as jax_terms
from celerite2 import terms, GaussianProcess
from exoplanet.distributions import estimate_inverse_gamma_parameters

from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
from matplotlib.lines import Line2D
import seaborn as sns

from volcano.utils import *

np.random.seed(42)
starry.config.lazy = False


def make_plots(
    lc_in,
    lc_eg,
    samples,
    yticks,
    ylim,
    xticks_in,
    xticks_eg,
    res_yticks,
    res_ylim,
    gp=True,
    cmap_norm=colors.Normalize(vmin=0.0, vmax=1500),
):
    #  Compute epheremis
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

    xo_in, yo_in, ro_in = get_occultor_position_and_radius(
        eph_io_in, eph_jup_in, occultor_is_jupiter=True
    )
    xo_eg, yo_eg, ro_eg = get_occultor_position_and_radius(
        eph_io_eg, eph_jup_eg, occultor_is_jupiter=True
    )

    # Phase
    theta_in = eph_io_in["theta"].value
    theta_eg = eph_io_eg["theta"].value

    # Fit single map model with different map amplitudes for ingress and egress
    ydeg_inf = 20
    map = starry.Map(ydeg_inf)

    # Evalute model on denser grid
    xo_in_dense = np.linspace(xo_in[0], xo_in[-1], 200)
    yo_in_dense = np.linspace(yo_in[0], yo_in[-1], 200)
    theta_in_dense = np.linspace(theta_in[0], theta_in[-1], 200)

    xo_eg_dense = np.linspace(xo_eg[0], xo_eg[-1], 200)
    yo_eg_dense = np.linspace(yo_eg[0], yo_eg[-1], 200)
    theta_eg_dense = np.linspace(theta_eg[0], theta_eg[-1], 200)

    t_in_dense = np.linspace(t_in[0], t_in[-1], 200)
    t_eg_dense = np.linspace(t_eg[0], t_eg[-1], 200)

    median_map_moll_in = get_median_map(ydeg_inf, samples["x_in"], nsamples=50)
    median_map_moll_eg = get_median_map(ydeg_inf, samples["x_eg"], nsamples=50)
    median_map_in = get_median_map(
        ydeg_inf,
        samples["x_in"],
        projection=None,
        theta=np.mean(theta_in),
        nsamples=50,
    )
    median_map_eg = get_median_map(
        ydeg_inf,
        samples["x_eg"],
        projection=None,
        theta=np.mean(theta_eg),
        nsamples=50,
    )

    median_map_moll_in = np.clip(median_map_moll_in, 0.1, 1e5)
    median_map_moll_eg = np.clip(median_map_moll_eg, 0.1, 1e5)
    median_map_in = np.clip(median_map_in, 0.1, 1e5)
    median_map_eg = np.clip(median_map_eg, 0.1, 1e5)

    # Make plot
    if gp:
        gp_pred_in = []
        gp_pred_eg = []
        gp_pred_in_dense = []
        gp_pred_eg_dense = []

        # Compute GP predictions
        for i in np.random.randint(0, len(samples), 100):
            # Ingress
            kernel_in = terms.Matern32Term(
                sigma=np.array(samples["sigma_gp"])[i][0],
                rho=np.array(samples["rho_gp"])[i][0],
            )
            gp = celerite2.GaussianProcess(
                kernel_in, t=t_in, mean=np.array(samples["flux_in"])[i]
            )
            gp.compute(t_in, yerr=(samples["f_err_in_mod"][i]))
            gp_pred_in.append(
                gp.predict(f_obs_in, t=t_in, include_mean=False)
                + samples["flux_in"][i]
            )
            gp_pred_in_dense.append(
                gp.predict(f_obs_in, t=t_in_dense, include_mean=False)
                + samples["flux_in_dense"][i]
            )
            # Egress
            kernel_eg = terms.Matern32Term(
                sigma=np.array(samples["sigma_gp"])[i][1],
                rho=np.array(samples["rho_gp"])[i][1],
            )
            gp = celerite2.GaussianProcess(
                kernel_eg, t=t_eg, mean=np.array(samples["flux_eg"])[i]
            )
            gp.compute(t_eg, yerr=(samples["f_err_eg_mod"][i]))
            gp_pred_eg.append(
                gp.predict(f_obs_eg, t=t_eg, include_mean=False)
                + samples["flux_eg"][i]
            )
            gp_pred_eg_dense.append(
                gp.predict(f_obs_eg, t=t_eg_dense, include_mean=False)
                + samples["flux_eg_dense"][i]
            )

    # Compute residuals
    f_in_median = np.median(samples["flux_in"], axis=0)
    f_eg_median = np.median(samples["flux_eg"], axis=0)

    if gp:
        f_in_median_gp = np.median(gp_pred_in, axis=0)
        f_eg_median_gp = np.median(gp_pred_eg, axis=0)

        res_in = f_obs_in - f_in_median_gp
        res_eg = f_obs_eg - f_eg_median_gp
    else:
        res_in = f_obs_in - f_in_median
        res_eg = f_obs_eg - f_eg_median

    # Set up the plot
    resol = 300
    nim = 7

    fig = plt.figure(figsize=(10, 9))
    fig.subplots_adjust(wspace=0.0)

    heights = [2, 4, 2]
    gs0 = fig.add_gridspec(
        nrows=1, ncols=2 * nim, bottom=0.71, left=0.05, right=0.98, hspace=0.4
    )
    gs1 = fig.add_gridspec(
        nrows=3,
        ncols=nim,
        height_ratios=heights,
        top=0.72,
        left=0.05,
        right=0.50,
        hspace=0.05,
    )
    gs2 = fig.add_gridspec(
        nrows=3,
        ncols=nim,
        height_ratios=heights,
        top=0.72,
        left=0.53,
        right=0.98,
        hspace=0.05,
    )

    # Maps
    ax_map_in = fig.add_subplot(gs0[0, :nim])
    ax_map_eg = fig.add_subplot(gs0[0, nim:])

    # Minimaps
    ax_im = [
        [fig.add_subplot(gs1[0, i]) for i in range(nim)],
        [fig.add_subplot(gs2[0, i]) for i in range(nim)],
    ]

    # Light curves
    ax_lc = [fig.add_subplot(gs1[1, :]), fig.add_subplot(gs2[1, :])]

    # Residuals
    ax_res = [fig.add_subplot(gs1[2, :]), fig.add_subplot(gs2[2, :])]

    # Plot maps
    cmap = "OrRd"
    map.show(
        image=median_map_moll_in,
        ax=ax_map_in,
        projection="Mollweide",
        norm=cmap_norm,
        colorbar=False,
        cmap=cmap,
    )
    map.show(
        image=median_map_moll_eg,
        ax=ax_map_eg,
        projection="Mollweide",
        norm=cmap_norm,
        colorbar=True,
        cmap=cmap,
    )
    
    ax_map_in.set_title("Ingress map\n", {lc_in.time[0].datetime.strftime("%Y-%m-%d %H:%M")})
    ax_map_eg.set_title("Egress map\n", {lc_eg.time[0].datetime.strftime("%Y-%m-%d %H:%M")})

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
                ro = ro_in
            else:
                map.show(
                    image=median_map_eg,
                    ax=a[n],
                    grid=False,
                    norm=cmap_norm,
                    cmap=cmap,
                )
                ro = ro_eg

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
    f_err_in_mod_median = np.median(samples["f_err_in_mod"], axis=0)
    f_err_eg_mod_median = np.median(samples["f_err_eg_mod"], axis=0)

    ax_lc[0].scatter(  # Data
        t_in,
        f_obs_in,
        color="black",
        marker="o",
        alpha=0.4,
    )

    if gp:
        #        for s in np.random.randint(0, len(samples["flux_in_dense"]), 10):
        #            ax_lc[0].plot(
        #                t_in_dense, samples["flux_in_dense"][s, :], "C0-", alpha=0.1
        #            )  # Model

        #  Plot full model
        for s in range(10):
            ax_lc[0].plot(
                t_in_dense, gp_pred_in_dense[s], "C1-", alpha=0.1
            )  # Model
    else:
        for s in np.random.randint(0, len(samples["flux_in_dense"]), 10):
            ax_lc[0].plot(
                t_in_dense, samples["flux_in_dense"][s, :], "C1-", alpha=0.1
            )  # Model

    # Residuals
    ax_res[0].errorbar(
        t_in,
        res_in,
        f_err_in_mod_median,
        color="black",
        marker="o",
        ecolor="black",
        linestyle="",
        alpha=0.4,
    )

    # Plot egress
    ax_lc[1].scatter(
        t_eg,
        f_obs_eg,
        color="black",
        marker="o",
        alpha=0.4,
    )

    if gp:
        #        for s in np.random.randint(0, len(samples["flux_eg_dense"]), 10):
        #            ax_lc[1].plot(
        #                t_eg_dense, samples["flux_eg_dense"][s, :], "C0-", alpha=0.1
        #            )  # Model

        #  Plot full model
        for s in range(10):
            ax_lc[1].plot(
                t_eg_dense, gp_pred_eg_dense[s], "C1-", alpha=0.1
            )  # Model
    else:
        for s in np.random.randint(0, len(samples["flux_eg_dense"]), 10):
            ax_lc[1].plot(
                t_eg_dense, samples["flux_eg_dense"][s, :], "C1-", alpha=0.1
            )  # Model

    # Residuals
    ax_res[1].errorbar(
        t_eg,
        res_eg,
        f_err_eg_mod_median,
        color="black",
        marker="o",
        ecolor="black",
        linestyle="",
        alpha=0.4,
    )

    # Legend
    #    if gp:
    #        custom_lines = [
    #            Line2D([0], [0], color="C1", linestyle="-"),
    #            Line2D([0], [0], color="C1", linestyle="dashed"),
    #        ]
    #        ax_lc[0].legend(custom_lines, ["Full model", "Excluding GP"])

    #  Ticks
    for a in ax_lc:
        a.set_xticklabels([])
        a.grid(alpha=0.5)
        a.set_yticks(yticks)
        a.set_ylim(ylim[0], ylim[1])

    for a in (ax_lc[0], ax_res[0]):
        a.set_xticks(xticks_in)
        a.set_xlim(left=-0.1)
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())

    for a in (ax_lc[1], ax_res[1]):
        a.set_xticks(xticks_eg)
        a.set_xlim(left=-0.1)
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.set_yticklabels([])

    for a in ax_res:
        a.grid(alpha=0.5)
        a.set_ylim(res_ylim)
        a.set_yticks(res_yticks)

    for j in range(2):
        ax_im[j][-1].set_zorder(-100)

    # Set common labels
    fig.text(0.5, 0.04, "Duration [minutes]", ha="center", va="center")
    ax_lc[0].set_ylabel("Intensity [GW/sr/um]")
    ax_res[0].set_ylabel("Residuals")

    year = lc_in.time[0].isot[:4]
    if gp:
        fig.savefig(f"irtf_{year}.pdf", bbox_inches="tight", dpi=500)
    else:
        fig.savefig(
            f"irtf_{year}_no_GP.pdf",
            bbox_inches="tight",
            dpi=500,
        )


# Plots for the the 1998 pair of light curves
with open("../../data/irtf_processed/lc_1998-08-27.pkl", "rb") as handle:
    lc_in = pkl.load(handle)

with open("../../data/irtf_processed/lc_1998-11-29.pkl", "rb") as handle:
    lc_eg = pkl.load(handle)

yticks = np.arange(0, 60, 10)
ylim = (-2, 52)
xticks_in = np.arange(0, 5, 1)
xticks_eg = np.arange(0, 6, 1)
res_yticks = np.arange(-2, 3, 1)
res_ylim = (-2.5, 2.5)

with open("scripts/irtf_1998_samples.pkl", "rb") as handle:
    samples = pkl.load(handle)
with open("scripts/irtf_1998_samples_no_GP.pkl", "rb") as handle:
    samples2 = pkl.load(handle)


def print_percentiles(samples, varname):
    mcmc = np.percentile(samples, [16, 50, 84])
    q = np.diff(mcmc)
    print(f"{varname}: {mcmc[1]:.3f} {q[0]:.3f} {q[1]:.3f}")


print("1998 event parameters:")
print_percentiles(samples["tau"], "tau")
print_percentiles(np.sqrt(samples["c2"]), "c")
print_percentiles(samples["amp_eg"], "a")
print_percentiles(samples["sigma_gp"][:, 0], "sigma_GP_I")
print_percentiles(samples["sigma_gp"][:, 1], "sigma_GP_E")
print_percentiles(samples["rho_gp"][:, 0], "rho_GP_I")
print_percentiles(samples["rho_gp"][:, 1], "rho_GP_E")
print_percentiles(samples["err_in_scale"], "err_scale_in")
print_percentiles(samples["err_eg_scale"], "err_scale_eg")
print_percentiles(np.exp(samples["ln_flux_offset"][:, 0]), "b_I")
print_percentiles(np.exp(samples["ln_flux_offset"][:, 1]), "b_E")


# Plot inferred maps and fit
make_plots(
    lc_in,
    lc_eg,
    samples,
    yticks,
    ylim,
    xticks_in,
    xticks_eg,
    res_yticks,
    res_ylim,
    cmap_norm=colors.LogNorm(vmin=30, vmax=1000),
)

# Model without GP
make_plots(
    lc_in,
    lc_eg,
    samples2,
    yticks,
    ylim,
    xticks_in,
    xticks_eg,
    res_yticks,
    res_ylim,
    gp=False,
    cmap_norm=colors.LogNorm(vmin=30, vmax=1000),
)


# Plots for the 2017 pair of light curves
with open("../../data/irtf_processed/lc_2017-03-31.pkl", "rb") as handle:
    lc_in = pkl.load(handle)

with open("../../data/irtf_processed/lc_2017-05-11.pkl", "rb") as handle:
    lc_eg = pkl.load(handle)

with open("scripts/irtf_2017_samples.pkl", "rb") as handle:
    samples = pkl.load(handle)

with open("scripts/irtf_2017_samples_no_GP.pkl", "rb") as handle:
    samples2 = pkl.load(handle)

print("2017 event parameters:")
print_percentiles(samples["tau"], "tau")
print_percentiles(np.sqrt(samples["c2"]), "c")
print_percentiles(samples["amp_eg"], "a")
print_percentiles(samples["sigma_gp"][:, 0], "sigma_GP_I")
print_percentiles(samples["sigma_gp"][:, 1], "sigma_GP_E")
print_percentiles(samples["rho_gp"][:, 0], "rho_GP_I")
print_percentiles(samples["rho_gp"][:, 1], "rho_GP_E")
print_percentiles(samples["err_in_scale"], "err_scale_in")
print_percentiles(samples["err_eg_scale"], "err_scale_eg")
print_percentiles(np.exp(samples["ln_flux_offset"][:, 0]), "b_I")
print_percentiles(np.exp(samples["ln_flux_offset"][:, 1]), "b_E")

yticks = np.arange(0, 100, 20)
ylim = (-2, 82)
xticks_in = np.arange(0, 5, 1)
xticks_eg = np.arange(0, 6, 1)
res_yticks = np.arange(-5, 7.5, 2.5)
res_ylim = (-5.5, 5.5)

# Model including a GP
make_plots(
    lc_in,
    lc_eg,
    samples,
    yticks,
    ylim,
    xticks_in,
    xticks_eg,
    res_yticks,
    res_ylim,
    cmap_norm=colors.LogNorm(vmin=30, vmax=1500),
)

# Model without GP
make_plots(
    lc_in,
    lc_eg,
    samples2,
    yticks,
    ylim,
    xticks_in,
    xticks_eg,
    res_yticks,
    res_ylim,
    gp=False,
    cmap_norm=colors.LogNorm(vmin=30, vmax=1500),
)