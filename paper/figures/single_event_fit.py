#!/usr/bin/env python
import os
import pickle as pkl
import sys

import matplotlib.gridspec as gridspec
import numpy as np
import pymc3 as pm
import starry
import theano
import theano.tensor as tt
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from volcano.utils import *

import exoplanet as xo

np.random.seed(42)
starry.config.lazy = True

# Load data
directory = "../../data/irtf_processed/"
lcs = []

for file in os.listdir(directory):
    with open(os.path.join(directory, file), "rb") as handle:
        lcs.append(pkl.load(handle))

# Select a single light curve
idx = 29
lc = lcs[idx]

f_obs = lc["flux"].value
f_err = lc["flux_err"].value


# Get ephemeris
eph_io = get_body_ephemeris(
    lc.time, body_id="501", step="1m", return_orientation=True
)
eph_jup = get_body_ephemeris(
    lc.time, body_id="599", step="1m", return_orientation=True
)


# Get occultor position
obl = eph_io["obl"]
inc = np.mean(eph_jup["inc"])
theta = np.array(eph_io["theta"])

xo_unrot, yo_unrot, zo, ro = get_occultor_position_and_radius(eph_io, eph_jup)

# Rotate to coordinate system where the obliquity of Io is 0
theta_rot = -obl.to(u.rad)
xo_rot, yo_rot = rotate_vectors(xo_unrot, yo_unrot, theta_rot)


# Pixel sampling model
ydeg_inf = 16
map = starry.Map(ydeg_inf)
lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
npix = Y2P.shape[0]

with pm.Model() as model:
    map = starry.Map(ydeg_inf)
    map.inc = inc.value

    # Uniform prior on the *pixels*
    p = pm.Dirichlet(
        "p",
        a=0.5 * np.ones(npix),
        shape=(npix,),
        testval=1e-05 * np.random.rand(npix),
    )

    ln_norm = pm.Normal("ln_norm", 2, 8, testval=np.log(100.0))
    x = tt.exp(ln_norm) * tt.dot(P2Y, p)

    map.amp = x[0]
    map[1:, :] = x[1:] / map.amp

    pm.Deterministic("amp", map.amp)
    pm.Deterministic("y1", map[1:, :])

    xo_offset = pm.Normal("xo_offset", 1.0, 0.1, testval=1.0)
    yo_offset = pm.Normal("yo_offset", 0.0, 0.01, testval=0.0)

    # Flux offset
    ln_flux_offset = pm.Normal("ln_flux_offset", 0.0, 4, testval=-2.0)

    # Compute flux
    _xo = theano.shared(xo_rot) + xo_offset
    _yo = theano.shared(yo_rot) + yo_offset

    flux = map.flux(xo=_xo, yo=_yo, ro=ro, theta=theta) + tt.exp(
        ln_flux_offset
    )
    pm.Deterministic("flux_pred", flux)

    sig = pm.Normal("sig", f_err[0], 0.1 * f_err[0], testval=f_err[0])
    pm.Normal("obs", mu=flux, sd=sig * tt.ones(len(f_obs)), observed=f_obs)


with model:
    start = xo.optimize(vars=model.vars[1:], options=dict(maxiter=9999))
    soln = xo.optimize(start=start, options=dict(maxiter=9999))

print(model.ndim)


# Evalute MAP model on denser grid
occ_phase_dense = np.linspace(lc["phase"][0], lc["phase"][-1], 200)
xo_dense = np.linspace(xo_rot[0], xo_rot[-1], 200)
yo_dense = np.linspace(yo_rot[0], yo_rot[-1], 200)
theta_dense = np.linspace(theta[0], theta[-1], 200)

with model:
    # Compute flux
    _xo = theano.shared(xo_dense) + xo_offset
    _yo = theano.shared(yo_dense) + yo_offset
    flux = map.flux(xo=_xo, yo=_yo, ro=ro, theta=theta_dense)
    map_flux_dense = xo.eval_in_model(flux, soln)


# SH model
PositiveNormal = pm.Bound(pm.Normal, lower=0.0)
ncoeff = (ydeg_inf + 1) ** 2

with pm.Model() as model_ylm:
    map = starry.Map(ydeg_inf)
    map.inc = inc.value

    cov = (3e-02) ** 2 * np.eye(ncoeff - 1)
    y1 = pm.MvNormal(
        "y1",
        np.zeros(ncoeff - 1),
        cov,
        shape=(ncoeff - 1,),
        testval=1e-03 * np.random.rand(ncoeff - 1),
    )  # testval=['y1'])
    ln_amp = pm.Normal("ln_amp", 0.0, 5.0, testval=np.log(soln["amp"]))
    pm.Deterministic("amp", tt.exp(ln_amp))

    map.amp = tt.exp(ln_amp)
    map[1:, :] = y1

    xo_offset = pm.Normal(
        "xo_offset", soln["xo_offset"], 1e-05, testval=soln["xo_offset"]
    )
    yo_offset = pm.Normal("yo_offset", soln["yo_offset"], 1e-05, testval=0.0)

    # Flux offset
    ln_flux_offset = pm.Normal("ln_flux_offset", 0.0, 4, testval=-2.0)

    # Compute flux
    _xo = theano.shared(xo_rot) + xo_offset
    _yo = theano.shared(yo_rot) + yo_offset

    flux = map.flux(xo=_xo, yo=_yo, ro=ro, theta=theta) + tt.exp(
        ln_flux_offset
    )
    pm.Deterministic("flux_pred", flux)
    sig = pm.Normal("sig", f_err[0], 0.1 * f_err[0], testval=f_err[0])
    pm.Normal("obs", mu=flux, sd=sig * tt.ones(len(f_obs)), observed=f_obs)

with model_ylm:
    start = xo.optimize(vars=model_ylm.vars[1:], options=dict(maxiter=9999))
    soln_ylm = xo.optimize(start=start, options=dict(maxiter=9999))


# Evaluate on dense grid
with model_ylm:
    # Compute flux
    _xo = theano.shared(xo_dense) + xo_offset
    _yo = theano.shared(yo_dense) + yo_offset
    flux = map.flux(xo=_xo, yo=_yo, ro=ro, theta=theta_dense)
    map_flux_dense_ylm = xo.eval_in_model(flux, soln_ylm)


def make_plot(
    map,
    ax_map,
    ax_im,
    ax_lc,
    ax_res,
    lc,
    soln,
    t_dense,
    map_flux_dense,
    residuals,
):
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
            ax=ax_im[n], theta=theta_im[n], res=resol, grid=False,
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
        ax_im[n].set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        ax_im[n].set_rasterization_zorder(0)

    # Plot light curve and fit
    ax_lc.errorbar(
        lc["phase"],
        lc["flux"].value,
        soln["sig"] * np.ones(len(lc["flux"])),
        color="black",
        fmt=".",
        ecolor="black",
        alpha=0.4,
    )
    ax_lc.plot(t_dense, map_flux_dense, "C1-")
    ax_lc.set_ylim(bottom=0.0)
    ax_lc.grid()
    ax_lc.set_ylabel("Flux [GW/um/sr]")
    ax_lc.set_xticklabels([])

    # Residuals
    ax_res.errorbar(
        lc["phase"],
        residuals,
        soln["sig"] * np.ones(len(lc["flux"])),
        color="black",
        fmt=".",
        ecolor="black",
        alpha=0.4,
    )
    ax_res.grid()
    ax_res.set(xlabel="Occultation phase", ylabel="Residuals")
    ax_res.xaxis.set_major_formatter(plt.FormatStrFormatter("%0.2f"))

    # Appearance
    for a in (ax_lc, ax_res):
        a.set_xticks(np.linspace(lc["phase"][0], lc["phase"][-1], nim))
    ax_im[-1].set_zorder(-100)


# Compute residuals
res = lc["flux"].value - soln["flux_pred"]
res_ylm = lc["flux"].value - soln_ylm["flux_pred"]

# Set up the plot
nim = 8

# Occultation params for mini subplots
xo_im = np.linspace(xo_rot[0], xo_rot[-1], nim) + soln["xo_offset"]
yo_im = np.linspace(yo_rot[0], yo_rot[-1], nim) + soln["yo_offset"]
theta_im = np.linspace(theta[0], theta[-1], nim)

# Initialize maps
map = starry.Map(ydeg_inf)
map.inc = inc.value
map.amp = soln["amp"]
map[1:, :] = soln["y1"]

map_ylm = starry.Map(ydeg_inf)
map_ylm.inc = inc.value
map_ylm.amp = soln_ylm["amp"]
map_ylm[1:, :] = soln_ylm["y1"]

# Initialize grid
fig = plt.figure(figsize=(14, 10))
heights = [3, 1, 3, 1]
gs1 = fig.add_gridspec(
    nrows=4, ncols=nim, left=0.05, right=0.48, height_ratios=heights
)
gs2 = fig.add_gridspec(
    nrows=4, ncols=nim, left=0.51, right=0.98, height_ratios=heights
)

# Add subplots
ax_map = []
ax_im = []
ax_lc = []
ax_res = []

for gs in (gs1, gs2):
    ax_map.append(fig.add_subplot(gs[0, :]))
    ax_im.append([fig.add_subplot(gs[1, i]) for i in range(nim)])
    ax_lc.append(fig.add_subplot(gs[2, :]))
    ax_res.append(fig.add_subplot(gs[3, :]))

make_plot(
    map_ylm,
    ax_map[0],
    ax_im[0],
    ax_lc[0],
    ax_res[0],
    lc,
    soln_ylm,
    occ_phase_dense,
    map_flux_dense_ylm,
    res_ylm,
)
make_plot(
    map,
    ax_map[1],
    ax_im[1],
    ax_lc[1],
    ax_res[1],
    lc,
    soln,
    occ_phase_dense,
    map_flux_dense,
    res,
)

ax_lc[1].set_ylabel("")
ax_res[1].set_ylabel("")

ax_map[0].set_title("Sparse prior on pixels", pad=20)
ax_map[1].set_title(
    r"Gaussian prior on $\mathrm{Y}_{lm}$ coefficients", pad=20
)

for a in ax_res:
    a.set_ylim(-5.4, 5.4)

# Save
fig.savefig("single_event_fit.pdf", bbox_inches="tight", dpi=500)
