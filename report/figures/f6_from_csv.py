"""F6 from the published table only: out/F6_spacetime.csv (cell type and our error at held-out INEL cells; no released values).

  python report/figures/f6_from_csv.py        (writes out/F6_spacetime.png and .pdf; the CSV is not rewritten)

make_figures.py fig_F6 draws the same two panels from the development cache and writes the CSV; this script lets anyone
redraw the figure from the repository alone. Panel b uses the same colour limit rule (99th percentile of |error|).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.dpi": 200,
    "pdf.fonttype": 42, "svg.hashsalt": "tfb", "axes.titleweight": "bold", "axes.titlelocation": "left",
})

d = pd.read_csv(OUT / "F6_spacetime.csv")
ns, nt = d.station_rank.max() + 1, d.step.max() + 1
ctype = np.zeros((ns, nt))
err = np.full((ns, nt), np.nan)
i, t = d.station_rank.to_numpy(), d.step.to_numpy()
inel = d.inel.astype(str).eq("True").to_numpy()
tg = d.scored_target.astype(str).eq("True").to_numpy()
ctype[i, t] = np.where(tg, 2, np.where(inel, 1, 0))
err[i, t] = np.where(inel, d.err_kmh.to_numpy(float), np.nan)

fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.1), sharex=True, gridspec_kw=dict(hspace=.3))
cm1 = ListedColormap(["#C9D1CB", "#0B6644", "#C98A1B"])
axes[0].imshow(ctype, aspect="auto", cmap=cm1, vmin=-.5, vmax=2.5, origin="lower", interpolation="nearest", extent=[0, 24, -.5, ns - .5])
axes[0].set_title("a  cell types: released (grey), held-out INEL (green), scored target (amber)", fontsize=6.6)
axes[0].set_ylabel("station\n(corridor order)", fontsize=6)
lim = np.nanpercentile(np.abs(err), 99)
cm2 = plt.get_cmap("PuOr_r").copy()
cm2.set_bad("#F2F2F2")
im = axes[1].imshow(np.ma.masked_invalid(err), aspect="auto", cmap=cm2, vmin=-lim, vmax=lim, origin="lower",
                    interpolation="nearest", extent=[0, 24, -.5, ns - .5])
axes[1].set_title("b  error at held-out INEL cells (ours - released)", fontsize=6.6)
axes[1].set_ylabel("station", fontsize=6)
axes[1].set_xlabel("hour of day")
axes[1].set_xticks(range(0, 25, 3))
cb = fig.colorbar(im, ax=axes, pad=.01, fraction=.03)
cb.set_label("km/h", fontsize=5.8)
cb.ax.tick_params(labelsize=5.5)
kw = dict(bbox_inches="tight", pad_inches=.03)
fig.savefig(OUT / "F6_spacetime.png", dpi=200, metadata={"Software": None}, **kw)
fig.savefig(OUT / "F6_spacetime.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None}, **kw)
e = err[np.isfinite(err)]
print(f"F6 {ns}x{nt}; INEL cells {int(inel.sum())}; RMSE {np.sqrt(np.mean(e ** 2)):.1f} km/h (published values are rounded to .1)")
