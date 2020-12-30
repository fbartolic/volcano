import numpy as np
import pickle as pkl

import starry
import jax.numpy as jnp
from jax import random, jit, vmap, lax
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import print_summary
from numpyro.infer import *

import celerite2.jax
from celerite2.jax import terms as jax_terms
from celerite2 import terms, GaussianProcess
from exoplanet.distributions import estimate_inverse_gamma_parameters

from astropy import units as u
from astropy.table import Table
from astropy.timeseries import TimeSeries

from volcano.utils import *

np.random.seed(42)
starry.config.lazy = False
numpyro.enable_x64()


def fit_model(ydeg_inf, lc_in, lc_eg):
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
    theta_in_dense = np.linspace(theta_in[0], theta_in[-1], 200)

    xo_eg_dense = np.linspace(xo_eg[0], xo_eg[-1], 200)
    yo_eg_dense = np.linspace(yo_eg[0], yo_eg[-1], 200)
    theta_eg_dense = np.linspace(theta_eg[0], theta_eg[-1], 200)

    t_dense_in = np.linspace(t_in[0], t_in[-1], 200)
    t_dense_eg = np.linspace(t_eg[0], t_eg[-1], 200)

    # Compute design matrices
    map = starry.Map(ydeg_inf)
    A_in = jnp.array(
        map.design_matrix(xo=xo_in, yo=yo_in, ro=ro_in, theta=theta_in)
    )
    A_eg = jnp.array(
        map.design_matrix(xo=xo_eg, yo=yo_eg, ro=ro_eg, theta=theta_eg)
    )
    A_in_dense = jnp.array(
        map.design_matrix(
            xo=xo_in_dense, yo=yo_in_dense, ro=ro_in, theta=theta_in_dense
        )
    )
    A_eg_dense = jnp.array(
        map.design_matrix(
            xo=xo_eg_dense, yo=yo_eg_dense, ro=ro_eg, theta=theta_eg_dense
        )
    )
    # Set the prior scale tau0 for the global scale parameter tau
    D = npix
    N = len(f_obs)
    peff = 0.8 * D  # Effective number of parameters
    sig = np.mean(f_err_in)
    tau0 = peff / (D - peff) * sig / np.sqrt(len(f_obs))
    print("tau0", tau0)

    # Other constants for the model
    slab_scale = 50.0
    slab_df = 10

    def model():
        #  Specify Finish Horseshoe prior
        # Non-centered distributions- loc=0, width=1 then shift/stretch afterwards
        beta_raw = numpyro.sample("beta_raw", dist.HalfNormal(1.0).expand([D]))
        lamda_raw = numpyro.sample(
            "lamda_raw", dist.HalfCauchy(1.0).expand([D])
        )
        tau_raw = numpyro.sample("tau_raw", dist.HalfCauchy(1.0))
        c2_raw = numpyro.sample(
            "c2_raw", dist.InverseGamma(0.5 * slab_df, 0.5 * slab_df)
        )

        # Do the shifting/stretching
        tau = numpyro.deterministic("tau", tau_raw * tau0)
        c2 = numpyro.deterministic("c2", slab_scale ** 2 * c2_raw)
        lamda_tilde = numpyro.deterministic(
            "lamda_tilde",
            jnp.sqrt(c2)
            * lamda_raw
            / jnp.sqrt(c2 + tau ** 2 * lamda_raw ** 2),
        )
        numpyro.deterministic(
            "mu_meff",
            (tau / sig * np.sqrt(N)) / (1 + tau / sig * np.sqrt(N)) * D,
        )

        # The Finnish Horseshoe prior on p
        p = numpyro.deterministic("p", tau * lamda_tilde * beta_raw)
        x = jnp.dot(P2Y, p)

        # Run the smoothing filter
        S = jnp.array(get_smoothing_filter(ydeg_inf, 2 / ydeg_inf))
        x_s = jnp.dot(S, x[:, None]).reshape(-1)

        # Allow for a different amplitude of the egress map
        amp_eg = numpyro.sample("amp_eg", dist.Normal(1.0, 0.1))
        numpyro.deterministic("x_in", x_s)
        numpyro.deterministic("x_eg", amp_eg * x_s)

        # Compute flux
        ln_flux_offset = numpyro.sample(
            "ln_flux_offset", dist.Normal(0.0, 4.0).expand([2])
        )

        flux_in = jnp.dot(A_in, x_s[:, None]).reshape(-1) + jnp.exp(
            ln_flux_offset[0]
        )
        flux_eg = jnp.dot(A_eg, amp_eg * x_s[:, None]).reshape(-1) + jnp.exp(
            ln_flux_offset[1]
        )

        numpyro.deterministic("flux_in", flux_in)
        numpyro.deterministic("flux_eg", flux_eg)
        flux = jnp.concatenate([flux_in, flux_eg])

        # Dense grid
        flux_in_dense = jnp.dot(A_in_dense, x_s[:, None]).reshape(-1)
        flux_eg_dense = jnp.dot(A_eg_dense, amp_eg * x_s[:, None]).reshape(-1)
        numpyro.deterministic("flux_in_dense", flux_in_dense)
        numpyro.deterministic("flux_eg_dense", flux_eg_dense)

        # Rescale errorbars
        ln_K = numpyro.sample("ln_K", dist.HalfNormal(0.05).expand([2]))

        # GP likelihood
        params = estimate_inverse_gamma_parameters(0.001, t_eg[-1])
        sigma = numpyro.sample(
            "sigma_gp", dist.HalfNormal(0.1 * f_err_in[0]).expand([2])
        )
        rho = numpyro.sample(
            "rho_gp",
            dist.InverseGamma(params["alpha"], params["beta"]).expand([2]),
        )
        kernel_in = jax_terms.Matern32Term(sigma=sigma[0], rho=rho[0])
        kernel_eg = jax_terms.Matern32Term(sigma=sigma[1], rho=rho[1])

        flux_in_fun = lambda _: flux_in
        flux_eg_fun = lambda _: flux_eg

        # Ingress GP
        gp_in = celerite2.jax.GaussianProcess(kernel_in, mean=flux_in_fun)
        gp_in.compute(
            t_in, diag=(jnp.exp(ln_K[0]) * f_err_in) ** 2, check_sorted=False
        )
        numpyro.sample("obs_in", gp_in.numpyro_dist(), obs=f_obs_in)

        # Egress GP
        gp_eg = celerite2.jax.GaussianProcess(kernel_eg, mean=flux_eg_fun)
        gp_eg.compute(
            t_eg, diag=(jnp.exp(ln_K[1]) * f_err_eg) ** 2, check_sorted=False
        )
        numpyro.sample("obs_eg", gp_eg.numpyro_dist(), obs=f_obs_eg)

    init_vals = {
        "beta_raw": 0.5 * jnp.ones(D),
        "lamda": jnp.ones(D),
        "tau_raw": 0.1,
        "c2_raw": 5 ** 2,
        "ln_flux_offset": -2 * np.ones(2),
        "ln_K": 1e-05 * np.ones(2),
        "sigma_gp": 0.001 * np.ones(2) * f_err_in[0],
        "rho_gp": 0.15 * np.ones(2),
    }

    nuts_kernel = NUTS(
        model,
        dense_mass=False,
        init_strategy=init_to_value(values=init_vals),
        target_accept_prob=0.95,
    )

    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key)
    samples = mcmc.get_samples()
    return samples


# Fit the 1998 pair of light curves
with open("../../data/irtf_processed/lc_1998-08-27.pkl", "rb") as handle:
    lc_in = pkl.load(handle)

with open("../../data/irtf_processed/lc_1998-11-29.pkl", "rb") as handle:
    lc_eg = pkl.load(handle)

samples = fit_model(25, lc_in, lc_eg)

with open("irtf_1998_samples.pkl", "wb") as handle:
    pkl.dump(samples, handle)

# Fit the 2017 pair of light curves
with open("../../data/irtf_processed/lc_2017-03-31.pkl", "rb") as handle:
    lc_in = pkl.load(handle)

with open("../../data/irtf_processed/lc_2017-05-11.pkl", "rb") as handle:
    lc_eg = pkl.load(handle)

samples2 = fit_model(25, lc_in, lc_eg)

with open("irtf_2017_samples.pkl", "wb") as handle:
    pkl.dump(samples2, handle)
