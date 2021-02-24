import numpy as np
import theano.tensor as tt
import pymc3 as pm
import starry
from starry._plotting import (
    get_moll_latitude_lines,
    get_moll_longitude_lines,
)

from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import optimize

np.random.seed(42)
starry.config.lazy = True

ydeg = 20
map = starry.Map(ydeg)
lat, lon, Y2P, P2Y, Dx, Dy = map.get_pixel_transforms(oversample=4)
npix = Y2P.shape[0]

std_p = 1.62

with pm.Model() as model:
    p = pm.Exponential("p", 1 / std_p, shape=(npix,))
    x = tt.dot(P2Y, p)
    pm.Deterministic("x", x)
    p_back = tt.dot(Y2P, x)
    pm.Deterministic("p_back", p_back)

    trace_pp = pm.sample_prior_predictive(10)


# Convert lat, lon to x,y coordinates in Mollewiede projection
def lon_lat_to_mollweide(lon, lat):
    lat *= np.pi / 180
    lon *= np.pi / 180

    f = lambda x: 2 * x + np.sin(2 * x) - np.pi * np.sin(lat)
    theta = optimize.newton(f, 0.3)

    x = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)

    return x, y


x_mol = np.zeros(npix)
y_mol = np.zeros(npix)

for idx, (lo, la) in enumerate(zip(lon, lat)):
    x_, y_ = lon_lat_to_mollweide(lo, la)
    x_mol[idx] = x_
    y_mol[idx] = y_


def plot_grid_lines(ax, alpha=0.6):
    """
    Code from https://github.com/rodluger/starry/blob/0546b4e445f6570b9a1cf6e33068e01a96ecf20f/starry/maps.py.
    """
    ax.axis("off")

    borders = []
    x = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), 10000)
    y = np.sqrt(2) * np.sqrt(1 - (x / (2 * np.sqrt(2))) ** 2)
    borders += [ax.fill_between(x, 1.1 * y, y, color="w", zorder=-1)]
    borders += [
        ax.fill_betweenx(0.5 * x, 2.2 * y, 2 * y, color="w", zorder=-1)
    ]
    borders += [ax.fill_between(x, -1.1 * y, -y, color="w", zorder=-1)]
    borders += [
        ax.fill_betweenx(0.5 * x, -2.2 * y, -2 * y, color="w", zorder=-1)
    ]

    x = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), 10000)
    a = np.sqrt(2)
    b = 2 * np.sqrt(2)
    y = a * np.sqrt(1 - (x / b) ** 2)
    borders = [None, None]
    (borders[0],) = ax.plot(x, y, "k-", alpha=1, lw=1.5)
    (borders[1],) = ax.plot(x, -y, "k-", alpha=1, lw=1.5)
    lats = get_moll_latitude_lines()
    latlines = [None for n in lats]
    for n, l in enumerate(lats):
        (latlines[n],) = ax.plot(
            l[0], l[1], "k-", lw=0.8, alpha=alpha, zorder=100
        )
    lons = get_moll_longitude_lines()
    lonlines = [None for n in lons]
    for n, l in enumerate(lons):
        (lonlines[n],) = ax.plot(
            l[0], l[1], "k-", lw=0.8, alpha=alpha, zorder=100
        )
    ax.fill_between(x, y, y + 10, color="white")
    ax.fill_between(x, -(y + 10), -y, color="white")


idx = -1
p_sample = trace_pp["p"][idx]
x_sample = trace_pp["x"][idx]
p_back_sample = trace_pp["p_back"][idx]

map = starry.Map(ydeg)
map.amp = x_sample[0]
map[1:, :] = x_sample[1:] / map.amp

px_sample = map.render(
    res=300,
    projection="Mollweide",
).eval()

# Normalize all pixels to same scale
norm = np.max(p_sample)
p_sample /= norm
px_sample /= norm
p_back_sample /= norm

fig, ax = plt.subplots(
    2, 2, figsize=(10, 6), gridspec_kw={"height_ratios": [4, 1]}
)
fig.subplots_adjust(wspace=0.15, top=0.5)

cmap = "OrRd"
vmax = np.max(p_sample)
order = np.argsort(p_sample)
im1 = ax[0, 0].scatter(
    x_mol[order],
    y_mol[order],
    s=15,
    c=p_sample[order],
    ec="none",
    cmap=cmap,
    marker="o",
    norm=colors.Normalize(vmin=0, vmax=1.0),
)

for a in ax[0, :].flatten():
    dx = 2.0 / 300
    extent = (
        -(1 + dx) * 2 * np.sqrt(2),
        2 * np.sqrt(2),
        -(1 + dx) * np.sqrt(2),
        np.sqrt(2),
    )
    a.axis("off")
    a.set_xlim(-2 * np.sqrt(2) - 0.05, 2 * np.sqrt(2) + 0.05)
    a.set_ylim(-np.sqrt(2) - 0.05, np.sqrt(2) + 0.05)

#  Plot Ylm map
resol = 300
map = starry.Map(ydeg)
map.amp = x_sample[0]
map[1:, :] = x_sample[1:] / map.amp
map.show(
    image=px_sample,
    ax=ax[0, 1],
    projection="Mollweide",
    norm=colors.Normalize(vmin=0.0, vmax=1.0),
    cmap=cmap,
)

for a in ax[0, :].flatten():
    a.set_aspect("equal")

cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.21])
fig.colorbar(im1, cax=cbar_ax)

# Plot grid lines
plot_grid_lines(ax[0, 0], alpha=0.3)

# Histograms of pixel values
ax[1, 0].hist(
    p_sample,
    bins="auto",
    alpha=0.8,
    color="black",
    histtype="step",
    lw=2.0,
    density=True,
)

ax[1, 1].hist(
    p_back_sample,
    bins="auto",
    alpha=0.8,
    color="black",
    histtype="step",
    lw=2.0,
    density=True,
)

for a in (ax[1, 0], ax[1, 1]):
    a.spines["right"].set_visible(False)
    a.spines["top"].set_visible(False)
    a.set_xlim(-0.05, 0.8)
    a.set_xticks(np.arange(0, 1.0, 0.2))
    a.set_yticks([])
    a.set_xlabel("Intensity distribution")
    a.spines["left"].set_visible(False)


ax[0, 0].set_title("Sphixels\n(draw from prior)")
ax[0, 1].set_title("Spherical harmonics\n ($l=20$)")

fig.text(
    0.485, 0.345, "$\;\;\,\mathbf{P}^\dagger$\n$\\longrightarrow$", fontsize=20
)

# Save
fig.savefig("sphixels_to_pixels.pdf", bbox_inches="tight", dpi=500)