import matplotlib as mpl


COLORS = {
    "black": "black",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "sky_blue": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#666666",
}


def apply_journal_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        # "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
         "font.sans-serif": ["Helvetica"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.2,
        "patch.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "legend.frameon": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
    })


def panel_label(label):
    return f"({label})"


apply_journal_style()
