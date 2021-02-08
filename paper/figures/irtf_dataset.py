import os
import pickle as pkl

import numpy as np
from astropy.table import vstack
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

np.random.seed(42)

# Load all light curves
directory = "../../data/irtf_processed/"
lcs = []
for file in os.listdir(directory):
    with open(os.path.join(directory, file), "rb") as handle:
        lcs.append(pkl.load(handle))
lcs_stacked = vstack(lcs, metadata_conflicts="silent")
lcs_stacked.sort()

## Plot stacked light curves
# fig, ax = plt.subplots(figsize=(6, 2))
# ax.plot(lcs_stacked.time.decimalyear, lcs_stacked["flux"], "k.", alpha=0.1)
# ax.set_ylabel("Flux [GW/um/sr]")
# ax.set_xlabel("time [years]")
# ax.set_xticks(np.arange(1996, 2024, 4))
# ax.set_yticks(np.arange(0, 250, 50))

# Plot maximum flux for each light curve
max_f = np.array([np.sort(lcs[i]["flux"].value)[-2] for i in range(len(lcs))])
yr = np.array([lc.time.decimalyear[0] for lc in lcs])
idx_sorted = np.argsort(yr)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(yr[idx_sorted], max_f[idx_sorted], "k.", alpha=0.6)
ax.set_ylabel("Maximum flux\n [GW/um/sr]")
ax.set_xlabel("Time [years]")
ax.set_xticks(np.arange(1996, 2024, 4))
ax.set_yticks(np.arange(0, 250, 50))
ax.grid()
plt.savefig("irtf_max_flux.pdf", bbox_inches="tight", dpi=500)

# Plot a few example light curves
fig, ax = plt.subplots(4, 3, figsize=(5, 6), sharex=True)
fig.subplots_adjust(wspace=0.4)

idcs = np.random.randint(0, len(lcs) - 1, size=3 * 4)

for a, idx in zip(ax.flatten(), idcs):
    t = (lcs[idx].time.mjd - lcs[idx].time.mjd[0]) * 24 * 60
    f = lcs[idx]["flux"].value

    #     a.errorbar(t, lcs[i]['flux'].value, lcs[i]['flux_err'].value, color='black',
    #                              marker='.',linestyle='',  alpha=0.3)
    a.plot(t, f, "k.", alpha=0.3)
    a.locator_params(nbins=8, axis="x")
    a.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    a.set_ylim(bottom=0)

fig.text(0.5, 0.01, "Duration [minutes]", ha="center")
fig.text(-0.03, 0.5, "Flux [GW/um/sr]", va="center", rotation="vertical")
plt.savefig("irtf_sample_lightcurves.pdf", bbox_inches="tight", dpi=500)