import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
import pickle as pkl
import astropy.units as u

np.random.seed(42)

# The 1998 pair of light curves
with open("../../data/irtf_processed/lc_1998-08-27.pkl", "rb") as handle:
    lc1_in = pkl.load(handle)

with open("../../data/irtf_processed/lc_1998-11-29.pkl", "rb") as handle:
    lc1_eg = pkl.load(handle)


## The 2017 pair of light curves
# with open("../../data/irtf_processed/lc_2017-03-31.pkl", "rb") as handle:
#    lc2_in = pkl.load(handle)
#
# with open("../../data/irtf_processed/lc_2017-05-11.pkl", "rb") as handle:
#    lc2_eg = pkl.load(handle)

fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

ax[0].plot(
    (lc1_in.time.mjd - lc1_in.time.mjd[0]) * u.d.to(u.min),
    lc1_in["flux"],
    "ko",
    alpha=0.4,
)
ax[1].plot(
    (lc1_eg.time.mjd - lc1_eg.time.mjd[0]) * u.d.to(u.min),
    lc1_eg["flux"],
    "ko",
    alpha=0.4,
)

ax[0].set(xticks=np.arange(0, 5, 1), yticks=np.arange(0, 60, 10))
ax[1].set(xticks=np.arange(0, 6, 1))

ax[0].set_title(lc1_in.time.datetime[0].strftime("%Y-%m-%d %H:%M"))
ax[1].set_title(lc1_eg.time.datetime[0].strftime("%Y-%m-%d %H:%M"))

ax[0].set(ylabel="Flux [ GW/um/sr]")


for a in ax:
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())

for a in ax.flatten():
    a.grid(alpha=0.5)

fig.text(0.5, -0.02, "Duration [minutes]", ha="center", va="center")
fig.savefig("irtf_1998_occultations.pdf", bbox_inches="tight", dpi=500)