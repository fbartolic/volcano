import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl

import seaborn as sns

with open("scripts/irtf_1998_samples.pkl", "rb") as handle:
    samples = pkl.load(handle)

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
fig.subplots_adjust(hspace=0.3)

for i in range(len(samples["f_err_in_mod"][0])):
    sns.distplot(
        samples["f_err_in_mod"][:, i],
        hist=False,
        norm_hist=True,
        kde_kws={"alpha": 0.7, "lw": 1},
        ax=ax[0],
    )
for i in range(len(samples["f_err_in_mod"][0])):
    sns.distplot(
        samples["f_err_eg_mod"][:, i],
        hist=False,
        norm_hist=True,
        kde_kws={"alpha": 0.7, "lw": 1},
        ax=ax[1],
    )

ax[0].set_xlabel("Ingress errorbars [GW/um/sr]")
ax[1].set_xlabel("Egress errorbars [GW/um/sr]")
fig.text(0.07, 0.5, "Posterior probability", va="center", rotation="vertical")

for a in ax:
    a.grid()
    a.set_xlim(-0.1, 2.0)
    a.set_ylabel("")

fig.savefig("irtf_1998_errorbars.pdf", bbox_inches="tight", dpi=500)
