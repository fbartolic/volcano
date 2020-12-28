import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pickle as pkl

import starry
import celerite2.jax
from celerite2.jax import terms as jax_terms
from celerite2 import terms, GaussianProcess
from exoplanet.distributions import estimate_inverse_gamma_parameters

from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries

from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator

from volcano.utils import *

np.random.seed(42)
starry.config.lazy = False


def get_pos_rot(eph_io, eph_jup, method=""):
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


def make_plots(
    lc_in_n,
    lc_eg_n,
    samples,
    yticks,
    ylim,
    xticks_in,
    xticks_eg,
    res_yticks,
    cmap_norm=colors.Normalize(vmin=0.0, vmax=1500),
):
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

    xo_in, yo_in, ro_in = get_pos_rot(eph_io_in, eph_jup_in)
    xo_eg, yo_eg, ro_eg = get_pos_rot(eph_io_eg, eph_jup_eg)

    # Phase
    theta_in = eph_io_in["theta"].value
    theta_eg = eph_io_eg["theta"].value

    # Fit single map model with different map amplitudes for ingress and egress
    ydeg_inf = 25
    map = starry.Map(ydeg_inf)

    # Evalute MAP model on denser grid
    xo_in_dense = np.linspace(xo_in[0], xo_in[-1], 200)
    yo_in_dense = np.linspace(yo_in[0], yo_in[-1], 200)
    theta_in_dense = np.linspace(theta_in[0], theta_in[-1], 200)

    xo_eg_dense = np.linspace(xo_eg[0], xo_eg[-1], 200)
    yo_eg_dense = np.linspace(yo_eg[0], yo_eg[-1], 200)
    theta_eg_dense = np.linspace(theta_eg[0], theta_eg[-1], 200)

    t_in_dense = np.linspace(t_in[0], t_in[-1], 200)
    t_eg_dense = np.linspace(t_eg[0], t_eg[-1], 200)

    median_map_moll_in, std_map_moll_in = get_median_map(
        ydeg_inf, samples["x_in"]
    )
    median_map_moll_eg, std_map_moll_eg = get_median_map(
        ydeg_inf, samples["x_eg"]
    )
    median_map_in, std_map_in = get_median_map(
        ydeg_inf, samples["x_in"], projection=None, theta=np.mean(theta_in)
    )
    median_map_eg, std_map_eg = get_median_map(
        ydeg_inf, samples["x_eg"], projection=None, theta=np.mean(theta_eg)
    )

    # Make plot
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
        gp.compute(t_in, yerr=(np.exp(samples["ln_K"][i][0]) * f_err_in) ** 2)
        gp_pred_in_dense.append(
            gp.predict(f_obs_in, t=t_in_dense, include_mean=False)
        )
        # Egress
        kernel_eg = terms.Matern32Term(
            sigma=np.array(samples["sigma_gp"])[i][1],
            rho=np.array(samples["rho_gp"])[i][1],
        )
        gp = celerite2.GaussianProcess(
            kernel_eg, t=t_eg, mean=np.array(samples["flux_eg"])[i]
        )
        gp.compute(t_eg, yerr=(np.exp(samples["ln_K"][i][1]) * f_err_eg) ** 2)
        gp_pred_eg_dense.append(
            gp.predict(f_obs_eg, t=t_eg_dense, include_mean=False)
        )

    # Compute residuals
    res_in = f_obs_in - np.median(samples["flux_in"], axis=0)
    res_eg = f_obs_eg - np.median(samples["flux_eg"], axis=0)

    # Errorbar rescaling
    K = np.exp(np.median(samples["ln_K"], axis=0))

    # Set up the plot
    resol = 300
    nim = 8

    fig = plt.figure(figsize=(9, 7))
    fig.subplots_adjust(wspace=0.0)

    heights = [2, 3, 1]
    gs0 = fig.add_gridspec(
        nrows=1, ncols=2 * nim, bottom=0.65, left=0.05, right=0.98, hspace=0.4
    )
    gs1 = fig.add_gridspec(
        nrows=3,
        ncols=nim,
        height_ratios=heights,
        top=0.67,
        left=0.05,
        right=0.50,
        hspace=0.08,
    )
    gs2 = fig.add_gridspec(
        nrows=3,
        ncols=nim,
        height_ratios=heights,
        top=0.67,
        left=0.53,
        right=0.98,
        hspace=0.08,
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
    map.show(
        image=median_map_moll_in,
        ax=ax_map_in,
        projection="Mollweide",
        norm=cmap_norm,
        colorbar=False,
    )
    map.show(
        image=median_map_moll_eg,
        ax=ax_map_eg,
        projection="Mollweide",
        norm=cmap_norm,
        colorbar=True,
    )
    ax_map_in.set_title(f"Ingress map\n {lc_in.time[0].isot[:19]}")
    ax_map_eg.set_title(f"Egress map\n {lc_eg.time[0].isot[:19]}")

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
                    image=median_map_in, ax=a[n], grid=False, norm=cmap_norm
                )
                ro = ro_in
            else:
                map.show(
                    image=median_map_eg, ax=a[n], grid=False, norm=cmap_norm
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
    ax_lc[0].errorbar(  # Data
        t_in,
        f_obs_in,
        K[0] * f_err_in,
        color="black",
        fmt=".",
        ecolor="black",
        alpha=0.4,
    )

    for s in np.random.randint(0, len(samples["flux_in_dense"]), 10):
        ax_lc[0].plot(
            t_in_dense, samples["flux_in_dense"][s, :], "C1-", alpha=0.2
        )  # Model

    # Residuals
    res_norm = np.median(np.std(np.concatenate([res_in, res_eg])))

    ax_res[0].errorbar(
        t_in,
        res_in / res_norm,
        K[0] * f_err_in / res_norm,
        color="black",
        fmt=".",
        ecolor="black",
        alpha=0.4,
    )

    for i in np.random.randint(0, len(gp_pred_in_dense), 10):
        ax_res[0].plot(t_in_dense, gp_pred_in_dense[i], "C1-", alpha=0.2)

    # Plot egress
    ax_lc[1].errorbar(
        t_eg,
        f_obs_eg,
        K[1] * f_err_eg,
        color="black",
        fmt=".",
        ecolor="black",
        alpha=0.4,
    )

    for s in np.random.randint(0, len(samples["flux_eg_dense"]), 10):
        ax_lc[1].plot(
            t_eg_dense, samples["flux_eg_dense"][s, :], "C1-", alpha=0.2
        )  # Model

    # Residuals
    ax_res[1].errorbar(
        t_eg,
        res_eg / res_norm,
        K[1] * f_err_eg / res_norm,
        color="black",
        fmt=".",
        ecolor="black",
        alpha=0.4,
    )

    for i in np.random.randint(0, len(gp_pred_in_dense), 10):
        ax_res[1].plot(t_eg_dense, gp_pred_eg_dense[i], "C1-", alpha=0.2)

    #  Ticks
    for a in ax_lc:
        a.set_xticklabels([])
        a.grid()
        a.set_yticks(yticks)
        a.set_ylim(ylim[0], ylim[1])

    for a in (ax_lc[0], ax_res[0]):
        a.set_xticks(xticks_in)
        a.set_xlim(left=-0.2)
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())

    for a in (ax_lc[1], ax_res[1]):
        a.set_xticks(xticks_eg)
        a.set_xlim(left=-0.2)
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.set_yticklabels([])

    for a in ax_res:
        a.grid()
        #     a.set_ylim(-3.1, 3.1)
        a.set_yticks(res_yticks)

    for j in range(2):
        ax_im[j][-1].set_zorder(-100)

    # Set common labels
    fig.text(0.5, 0.04, "Duration [minutes]", ha="center", va="center")
    ax_lc[0].set_ylabel("Flux [GW/sr/um]")
    ax_res[0].set_ylabel("Residuals\n (norm.)")

    year = lc_in.time[0].isot[:4]
    fig.savefig(f"irtf_ingress_egress_{year}.pdf", bbox_inches="tight", dpi=500)

    # Plot inferred hot spot on top of Galileo maps
    combined_mosaic = Image.open("Io_SSI_VGR_bw_Equi_Clon180_8bit.tif")

    # The loaded map is [0,360] while Starry expects [-180, 180]
    galileo_map = np.roll(combined_mosaic, int(11445 / 2), axis=1)

    median_map_rect_in, std_map_rect_in = get_median_map(
        ydeg_inf, samples["x_in"], projection="Rectangular"

    )
    fig, ax = plt.subplots(figsize=(8, 4))
    map.show(
        image=median_map_rect_in, projection="rectangular",  ax=ax, colorbar=False,

    )
    ax.imshow(galileo_map, extent=(-180, 180, -90, 90), alpha=0.7, cmap='gray')
    ax.set_yticks(np.arange(-15, 60, 15));
    ax.set_xticks(np.arange(15, 105, 15));
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set(xlim=(15, 90), ylim=(-15, 45))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.savefig(f"irtf_spot_overlay_{year}.pdf", bbox_inches="tight", dpi=500)

# Plots for the the 1998 pair of light curves
lc_in_numpy = np.load("irtf_1998_ingress.npy")
lc_eg_numpy = np.load("irtf_1998_egress.npy")

yticks = np.arange(0, 60, 10)
ylim = (-2, 52)
xticks_in = np.arange(0, 4.5, 0.5)
xticks_eg = np.arange(0, 6.0, 0.5)
res_yticks = [-3.0, 0.0, 3.0]

with open("irtf_1998_samples.pkl", "rb") as handle:
    samples = pkl.load(handle)

make_plots(
    lc_in_numpy,
    lc_eg_numpy,
    samples,
    yticks,
    ylim,
    xticks_in,
    xticks_eg,
    res_yticks,
    cmap_norm=colors.Normalize(vmin=0, vmax=1200),
)

# Plots for the 2017 pair of light curves
lc_in_numpy2 = np.load("irtf_2017_ingress.npy")
lc_eg_numpy2 = np.load("irtf_2017_egress.npy")

with open("irtf_2017_samples.pkl", "rb") as handle:
    samples2 = pkl.load(handle)

yticks = np.arange(0, 100, 20)
ylim = (-2, 82)
xticks_in = np.arange(0, 4.5, 0.5)
xticks_eg = np.arange(0, 5.5, 0.5)
res_yticks = [-3.0, 0.0, 3.0]

make_plots(
    lc_in_numpy2,
    lc_eg_numpy2,
    samples2,
    yticks,
    ylim,
    xticks_in,
    xticks_eg,
    res_yticks,
    cmap_norm=colors.Normalize(vmin=0, vmax=2100),
)
