#!/usr/bin/env python
import numpy as np
import starry
import astropy.units as u
from astropy.time import Time
import sys
import os
import scipy
from scipy.linalg import cho_factor, cho_solve

from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from volcano.utils import get_body_ephemeris, get_body_vectors

np.random.seed(42)
starry.config.lazy = False


## Get 10 1hr observations of Io in eclipse/sunlight with 1sec cadence
# start = Time("2009-01-01", format="isot")
# stop = Time("2010-01-01", format="isot")
# start_times = np.random.uniform(start.mjd, stop.mjd, 100)
#
# eph_phase_sun = []
# eph_phase_em = []
#
## This takes a while
# nlightcurves = 30
## Phase curves in sunlight
# i = 0
# while len(eph_phase_sun) < nlightcurves:
#    t = start_times[i]
#    t_end = t + 2
#    npts = int((t_end - t) * 24 * 60 * 60)  # 1sec cadence
#    times = Time(np.linspace(t, t_end, npts), format="mjd")
#
#    eph = get_body_ephemeris(
#        times, body_id="501", step="1m", return_orientation=True
#    )
#
#    # Get only phase curves in sunlight
#    mask_sun = np.all(
#        [~eph["ecl_tot"], ~eph["ecl_par"], ~eph["occ_umbra"], ~eph["occ_sun"]],
#        axis=0,
#    )
#
#    if mask_sun.sum() > 100.0:
#        eph_phase_sun.append(eph[mask_sun][:100])
#    i = i + 1
#
## Phase curves in eclipse
# i = 0
# while len(eph_phase_em) < nlightcurves:
#    t = start_times[i]
#    t_end = t + 2
#    npts = int((t_end - t) * 24 * 60 * 60)  # 1sec cadence
#    times = Time(np.linspace(t, t_end, npts), format="mjd")
#
#    eph = get_body_ephemeris(
#        times, body_id="501", step="1m", return_orientation=True
#    )
#
#    # Get phase curves in eclipse
#    mask_ecl = np.all(
#        [eph["ecl_tot"], ~eph["occ_umbra"]],
#        axis=0,
#    )
#
#    if mask_ecl.sum() > 100.0:
#        eph_phase_em.append(eph[mask_ecl][:100])
#    i = i + 1
#
# eph_io_occ_sun = []
# eph_jup_occ_sun = []
# eph_io_occ_em = []
# eph_jup_occ_em = []
#
## Occultations in sunlight
# i = 0
# while len(eph_io_occ_sun) < nlightcurves:
#    t = start_times[i]
#    t_end = t + 2.0
#    npts = int((t_end - t) * 24 * 60 * 60)  # 1sec cadence
#    times = Time(np.linspace(t, t_end, npts), format="mjd")
#
#    eph_io = get_body_ephemeris(
#        times, body_id="501", step="1m", return_orientation=True
#    )
#
#    eph_jup = get_body_ephemeris(
#        times, body_id="599", step="1m", return_orientation=False
#    )
#
#    # Get occultations in sunlight
#    mask = eph_io["occ_sun"]
#
#    if mask.sum() > 0.0:
#        eph_io_occ_sun.append(eph_io[mask])
#        eph_jup_occ_sun.append(eph_jup[mask])
#
#    i = i + 1
#
## Occultations in eclipse
# i = 0
# while len(eph_io_occ_em) < nlightcurves:
#    t = start_times[i]
#    t_end = t + 2.0
#    npts = int((t_end - t) * 24 * 60 * 60)  # 1sec cadence
#    times = Time(np.linspace(t, t_end, npts), format="mjd")
#
#    eph_io = get_body_ephemeris(
#        times, body_id="501", step="1m", return_orientation=True
#    )
#
#    eph_jup = get_body_ephemeris(
#        times, body_id="599", step="1m", return_orientation=False
#    )
#
#    # Get occultations in eclipse
#    mask = eph_io["occ_umbra"]
#
#    if mask.sum() > 0.0:
#        eph_io_occ_em.append(eph_io[mask])
#        eph_jup_occ_em.append(eph_jup[mask])
#
#    i = i + 1
#
#
# def compute_design_matrix_phase(eph_list, reflected=False):
#    ydeg = 20
#    map = starry.Map(ydeg=ydeg, reflected=reflected)
#
#    A_ = []
#
#    for i in range(len(eph_list)):
#        eph = eph_list[i]
#
#        # Fix obliquity per light curve, this is fine because
#        # obliquity and inclination vary on timescales of years
#        obl = np.mean(eph["obl"])
#        inc = np.mean(eph["inc"])
#        map.obl = obl
#        map.inc = inc
#
#        theta = np.array(eph["theta"])
#        xs = np.array(eph["xs"])
#        ys = np.array(eph["ys"])
#        zs = np.array(eph["zs"])
#
#        # Compute the design matrix for a 1000 points per observing interval
#        if reflected is True:
#            m = map.design_matrix(theta=theta, xs=xs, ys=ys, zs=zs)
#
#        else:
#            m = map.design_matrix(theta=theta)
#
#        A_.append(m)
#
#    # Design matrix
#    return np.concatenate(A_)
#
#
# def compute_design_matrix_occ(
#    eph_occulted_list, eph_occultor_list, reflected=False
# ):
#    ydeg = 20
#    map = starry.Map(ydeg=ydeg, reflected=reflected)
#
#    A_ = []
#
#    for i in range(len(eph_occultor_list)):
#        eph_occulted = eph_occulted_list[i]
#        eph_occultor = eph_occultor_list[i]
#
#        obl = np.mean(eph_occulted["obl"])
#        inc = np.mean(eph_occulted["inc"])
#        map.obl = obl
#        map.inc = inc
#
#        theta = np.array(eph_occulted["theta"])
#        xs = np.array(eph_occulted["xs"])
#        ys = np.array(eph_occulted["ys"])
#        zs = np.array(eph_occulted["zs"])
#
#        # Convert everything to units where the radius of Io = 1
#        radius_occultor = eph_occultor["ang_width"] / eph_occulted["ang_width"]
#        rel_ra = (eph_occultor["RA"] - eph_occulted["RA"]).to(u.arcsec) / (
#            0.5 * eph_occulted["ang_width"].to(u.arcsec)
#        )
#        rel_dec = (eph_occultor["DEC"] - eph_occulted["DEC"]).to(u.arcsec) / (
#            0.5 * eph_occulted["ang_width"].to(u.arcsec)
#        )
#
#        xo = -rel_ra
#        yo = rel_dec
#        zo = np.ones(len(yo))
#        ro = np.mean(radius_occultor)
#
#        if reflected is True:
#            m = map.design_matrix(
#                theta=theta, xs=xs, ys=ys, zs=zs, xo=xo, yo=yo, zo=zo, ro=ro
#            )
#        else:
#            m = map.design_matrix(theta=theta, xo=xo, yo=yo, zo=zo, ro=ro)
#
#        A_.append(m)
#
#    # Remove parts when Io is behind jupiter
#    A = np.concatenate(A_)
#    y = np.zeros(int((ydeg + 1) ** 2))
#    y[0] = 1.0
#    f = A @ y[:, None]
#    mask = (f == 0).flatten()
#
#    return A[~mask, :]
#
#
## Phase curve emitted light
# A_phase_em = compute_design_matrix_phase(eph_phase_em, reflected=False)
#
## Phase curves in sunlight
# A_phase_sun = compute_design_matrix_phase(eph_phase_sun, reflected=False)
#
## Occultations in emitted light
# A_occ_em = compute_design_matrix_occ(
#    eph_io_occ_em, eph_jup_occ_em, reflected=False
# )
#
## Occultations in sunlight
# A_occ_sun = compute_design_matrix_occ(
#    eph_io_occ_sun, eph_jup_occ_sun, reflected=False
# )
#
## Combined phase curves
# A_phase = np.concatenate([A_phase_em, A_phase_sun])
#
## Combined occultations
# A_occ = np.concatenate([A_occ_em, A_occ_sun])
#
## Occultations + phase curves
# A_occ_phase = np.concatenate([A_phase, A_occ])
#
## Search for all mutual occultations of Galilean moons in a single year
# search_windows = np.linspace(start.mjd, stop.mjd, 200)
#
#
# def find_occultations(eph_io, eph_occultor):
#    # Convert everything to units where the radius of Io = 1
#    radius_occultor = eph_occultor["ang_width"] / eph_io["ang_width"]
#    rel_ra = (eph_occultor["RA"] - eph_io["RA"]).to(u.arcsec) / (
#        0.5 * eph_io["ang_width"].to(u.arcsec)
#    )
#    rel_dec = (eph_occultor["DEC"] - eph_io["DEC"]).to(u.arcsec) / (
#        0.5 * eph_io["ang_width"].to(u.arcsec)
#    )
#
#    xo = -rel_ra
#    yo = rel_dec
#    ro = np.mean(radius_occultor)
#
#    eps = 0.1
#    mask = np.sqrt(xo ** 2 + yo ** 2) < (ro + 1 + eps)
#
#    return mask
#
#
## Store ephemeris
# mut_eur_sun = []
# mut_gan_sun = []
# mut_cal_sun = []
#
# mut_eur_em = []
# mut_gan_em = []
# mut_cal_em = []
#
## Iterate over each time window, get ephemeris for all moons and search
## for occultations, this takes a few hours
# for i in range(len(search_windows) - 1):
#    t_start = search_windows[i]
#    t_end = search_windows[i + 1]
#    npts = int((t_end - t_start) * 24 * 60 * 60)  # 1sec cadence
#
#    times = Time(np.linspace(t_start, t_end, npts), format="mjd")
#
#    eph_io = get_body_ephemeris(
#        times, body_id="501", step="1m", return_orientation=True
#    )
#    eph_eur = get_body_ephemeris(
#        times, body_id="502", step="1m", return_orientation=False
#    )
#    eph_gan = get_body_ephemeris(
#        times, body_id="503", step="1m", return_orientation=False
#    )
#    eph_cal = get_body_ephemeris(
#        times, body_id="504", step="1m", return_orientation=False
#    )
#
#    # Select times when Io is not occulted by Jupiter and not in eclipse
#    mask = np.all(
#        [eph_io["occ_umbra"], eph_io["occ_sun"], eph_io["ecl_par"]], axis=0
#    )
#
#    eph_io = eph_io[~mask]
#    eph_eur = eph_eur[~mask]
#    eph_gan = eph_gan[~mask]
#    eph_cal = eph_cal[~mask]
#
#    # Find occultations with each of the moons
#    mask_eur = find_occultations(eph_io, eph_eur)
#    mask_gan = find_occultations(eph_io, eph_gan)
#    mask_cal = find_occultations(eph_io, eph_cal)
#
#    if mask_eur.sum() > 0:
#        mask_ecl = eph_io[mask_eur]["ecl_tot"] > 0.0
#        # Split reflected and emitted light ephemeris
#        if np.all(mask_ecl):
#            mut_eur_em.append(
#                [eph_io[mask_eur][mask_ecl], eph_eur[mask_eur][mask_ecl]]
#            )
#        else:
#            mut_eur_sun.append([eph_io[mask_eur], eph_eur[mask_eur]])
#    if mask_gan.sum() > 0:
#        mask_ecl = eph_io[mask_gan]["ecl_tot"] > 0.0
#        if np.all(mask_ecl):
#            mut_gan_em.append(
#                [eph_io[mask_gan][mask_ecl], eph_gan[mask_gan][mask_ecl]]
#            )
#        else:
#            mut_gan_sun.append([eph_io[mask_gan], eph_gan[mask_gan]])
#    if mask_cal.sum() > 0:
#        mask_ecl = eph_io[mask_cal]["ecl_tot"] > 0.0
#        if np.all(mask_ecl):
#            mut_cal_em.append(
#                [eph_io[mask_cal][mask_ecl], eph_cal[mask_cal][mask_ecl]]
#            )
#        else:
#            mut_cal_sun.append([eph_io[mask_cal], eph_cal[mask_cal]])
#
## Europa
# A_mut_eur_em = compute_design_matrix_occ(
#    [eph[0] for eph in mut_eur_em],
#    [eph[1] for eph in mut_eur_em],
#    reflected=False,
# )
#
# A_mut_eur_sun = compute_design_matrix_occ(
#    [eph[0] for eph in mut_eur_sun],
#    [eph[1] for eph in mut_eur_sun],
#    reflected=False,
# )
#
## Ganymede
## A_mut_gan_em = compute_design_matrix_occ([eph[0] for eph in mut_gan_em],
##                                          [eph[1] for eph in mut_gan_em],
##                                          reflected=False)
# A_mut_gan_sun = compute_design_matrix_occ(
#    [eph[0] for eph in mut_gan_sun],
#    [eph[1] for eph in mut_gan_sun],
#    reflected=False,
# )
#
## Callisto
## A_mut_cal_em = compute_design_matrix_occ([eph[0] for eph in mut_cal_em],
##                                          [eph[1] for eph in mut_cal_em],
##                                          reflected=False)
# A_mut_cal_sun = compute_design_matrix_occ(
#    [eph[0] for eph in mut_cal_sun],
#    [eph[1] for eph in mut_cal_sun],
#    reflected=False,
# )
#
# A_mut = np.concatenate(
#    [A_mut_eur_sun, A_mut_gan_sun, A_mut_cal_sun, A_mut_eur_em]
# )
#
# directory = "out"
#
# if not os.path.exists(directory):
#    os.makedirs(directory)
#
## Save design matrices to file
# np.save(os.path.join(directory, "A_phase.npy"), A_phase)
# np.save(os.path.join(directory, "A_occ.npy"), A_occ)
# np.save(os.path.join(directory, "A_occ_phase"), A_occ_phase)
# np.save(os.path.join(directory, "A_mut.npy"), A_mut)

# Load previously computed design matrices
A_phase = np.load("out/A_phase.npy")
A_occ = np.load("out/A_occ.npy")
A_occ_phase = np.load("out/A_occ_phase.npy")
A_mut = np.load("out/A_mut.npy")


def compute_posterior_shrinkage(A, avg_across_m=True, snr=100):
    """
    Computes the posterior covariance matrix for a given design matrix
    and data covariance matrix and returns the posterio shrinkage avaraged
    across coefficients with different m-s. The posterior shrinkage in this case
    is defined as 1 - sigma_post^2/sigma_prior^2 where sigma_post^2 and
    sigma_prior^2 are the entries on the diagonal over the posterior and
    prior covariance matrices respectively.
    """
    ncoeff = len(A[0, :])
    ydeg = int(np.sqrt(ncoeff) - 1)

    # Compute posterior covariance
    L = 1e4
    cho_C = starry.linalg.solve(
        design_matrix=A,
        data=np.random.randn(A.shape[0]),
        C=(0.1 * np.ones_like(A.shape[0])) ** 2,
        L=L,
        N=ncoeff,
    )[1]
    S = 1 - np.diag(cho_C @ cho_C.T) / L

    # Average across m
    S_mean = np.zeros(ydeg + 1)

    if avg_across_m:
        start = 0
        for l in range(ydeg + 1):
            S_mean[l] = np.mean(S[start : int(start + 2 * l + 1)])
            start += 2 * l + 1
        return S_mean

    else:
        ls = np.floor(np.sqrt(np.arange(int((ydeg + 1) ** 2))))

        return ls


s_phase = compute_posterior_shrinkage(A_phase[:, :])
s_occ = compute_posterior_shrinkage(A_occ[:, :])
s_occ_phase = compute_posterior_shrinkage(A_occ_phase[::2, :])
s_mut = compute_posterior_shrinkage(A_mut[:, :])


fig, ax = plt.subplots(figsize=(7, 4))

lcut = 16

(p1,) = ax.plot(s_phase[:lcut], "C0.-", label="Phase curves")
(p3,) = ax.plot(s_occ[:lcut], "C1.-", label="Occultations by Jupiter")
(p5,) = ax.plot(
    s_occ_phase[:lcut], "C2.-", label="Occultations by Jupiter + phase curves"
)
(p7,) = ax.plot(s_mut[:lcut], "C3.-", label="Mutual occultations")
ax.legend(prop={"size": 10}, loc="upper right", bbox_to_anchor=(1.0, 0.95))

ax.set_xlabel(r"Map degree $l$")
ax.set_ylabel("Posterior shrinkage")
ax.grid(alpha=0.5)
ax.set_ylim(-0.03, 1.03)
ax.set_xlim(-0.5, lcut - 0.5)
ax.set_xticks(np.arange(lcut))

plt.savefig("information_content.pdf", bbox_inches="tight")