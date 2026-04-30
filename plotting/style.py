import os
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    HAS_MPLHEP = True
except ImportError:
    HAS_MPLHEP = False


MET_STYLES = {
    "l1": {
        "label": "L1 MET",
        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-",
    },
    "ml": {
        "label": "ML MET",
        "color": "tab:red",
        "marker": "s",
        "linestyle": "-",
    },
    "gen": {
        "label": "Gen MET",
        "color": "black",
        "marker": None,
        "linestyle": "--",
    },
}


def setup_style():
    """
    Minimal CMS Phase-2 style for MET plots.
    """
    if HAS_MPLHEP:
        plt.style.use(hep.style.CMS)

    plt.rcParams["figure.figsize"] = (7.0, 6.0)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 200

    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fontsize"] = 13

    plt.rcParams["lines.linewidth"] = 2.5
    plt.rcParams["lines.markersize"] = 6.5


def add_phase2_label(ax=None, energy="14 TeV", pu=200):
    """
    Add:
      CMS Phase-2 Simulation
      14 TeV, PU200
    """
    if ax is None:
        ax = plt.gca()

    right_text = f"{energy}, PU{pu}"

    if HAS_MPLHEP:
        hep.cms.text("Phase-2 Simulation", ax=ax, loc=0)
        ax.text(
            1.0,
            1.01,
            right_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
        )
    else:
        ax.text(
            0.0,
            1.01,
            r"$\bf{CMS}$ Phase-2 Simulation",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=15,
        )
        ax.text(
            1.0,
            1.01,
            right_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
        )


def get_met_style(kind):
    """
    kind: 'l1', 'ml', or 'gen'
    """
    if kind not in MET_STYLES:
        raise ValueError(f"Unknown MET style: {kind}")

    return MET_STYLES[kind]


def apply_axis_style(
    ax,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    logy=False,
    grid=False,
):
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if logy:
        ax.set_yscale("log")

    ax.grid(grid, alpha=0.3)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )


def save_figure(fig, outpath, formats=("png", "pdf")):
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    root, ext = os.path.splitext(outpath)

    if ext:
        fig.savefig(outpath, bbox_inches="tight")
        print(f"[plot] Saved: {outpath}")
        return

    for fmt in formats:
        path = f"{outpath}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        print(f"[plot] Saved: {path}")