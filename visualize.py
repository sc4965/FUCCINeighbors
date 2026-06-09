"""Visualize a FUCCI-4 / VME pipeline results CSV.

Standalone (pandas + numpy + matplotlib only) so it can run anywhere on the
``results.csv`` produced by ``pipeline.py`` -- no need to re-run the pipeline.

Focus: the mitotic (M) population inside the Virus Microenvironment (VME).
  * proportion of mitotic cells over time, VME vs control
  * what happens to mitotic cells -- divide vs. arrest vs. exit (using lineage
    columns) and the phase they transition into after M
  * per-cell phase "ribbons" to see whether mitotic VME cells follow one path or
    diverge

Usage
-----
    python3 visualize.py --csv results.csv --out-prefix viz
    python3 visualize.py --csv results.csv --show          # interactive

Every panel is written to ``<out-prefix>_<name>.png`` (and/or shown).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("fucci_vme.visualize")

# Discrete, colour-blind-friendly palette for FUCCI-4 phases. "M" is deliberately
# a vivid magenta so the mitotic population pops out of every plot.
PHASE_ORDER = ["G0", "G1", "G1/S", "S", "G2/M", "M", "undetermined"]
PHASE_COLORS = {
    "G0": "#9e9e9e",
    "G1": "#d62728",
    "G1/S": "#ff7f0e",
    "S": "#1f77b4",
    "G2/M": "#2ca02c",
    "M": "#d000d0",
    "undetermined": "#dddddd",
}
MITOTIC_PHASE = "M"


# --------------------------------------------------------------------------- #
# Loading / small helpers
# --------------------------------------------------------------------------- #
def load_results(csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    for col in ("is_vme", "is_infected", "is_index"):
        if col in df.columns:
            df[col] = df[col].astype(bool)
    if "phase" not in df.columns:
        raise KeyError("CSV has no 'phase' column - was the pipeline run completed?")
    if MITOTIC_PHASE not in set(df["phase"].unique()):
        logger.warning(
            "No '%s' phase present. Re-run the pipeline with --detect-mitosis to "
            "label mitotic cells; mitosis-specific panels will be empty.",
            MITOTIC_PHASE,
        )
    return df


def _time_col(df: pd.DataFrame) -> str:
    return "frame_since_infection" if "frame_since_infection" in df.columns else "frame"


def _phases_present(df: pd.DataFrame) -> list[str]:
    present = [p for p in PHASE_ORDER if p in set(df["phase"].unique())]
    extras = [p for p in df["phase"].unique() if p not in PHASE_ORDER]
    return present + sorted(map(str, extras))


def _new_axes(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, plt, out_prefix, name, show):
    if out_prefix is not None:
        path = f"{out_prefix}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Wrote %s", path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# 1. Mitotic fraction over time (the headline question)
# --------------------------------------------------------------------------- #
def plot_mitotic_fraction(df, out_prefix=None, show=False):
    plt = _new_axes(show)
    tcol = _time_col(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    for is_vme, label, color in (
        (True, "VME (neighbors)", "#d000d0"),
        (False, "Control (distal)", "#555555"),
    ):
        grp = df[df["is_vme"] == is_vme]
        if grp.empty:
            continue
        frac = (
            grp.assign(is_m=grp["phase"].eq(MITOTIC_PHASE))
            .groupby(tcol)["is_m"]
            .agg(["mean", "count"])
            .reset_index()
        )
        ax.plot(frac[tcol], frac["mean"], color=color, lw=2, label=label)
        # Wilson-ish SEM band for a proportion
        p = frac["mean"].to_numpy()
        n = frac["count"].to_numpy().clip(min=1)
        sem = np.sqrt(np.clip(p * (1 - p), 0, None) / n)
        ax.fill_between(frac[tcol], p - sem, p + sem, color=color, alpha=0.2)

    if tcol == "frame_since_infection":
        ax.axvline(0, color="black", ls="--", lw=1, label="infection")
    ax.set_xlabel(tcol.replace("_", " "))
    ax.set_ylabel("Fraction of cells in M (mitotic)")
    ax.set_title("Proportion of mitotic cells over time")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, plt, out_prefix, "mitotic_fraction", show)
    return fig


# --------------------------------------------------------------------------- #
# 2. Phase composition over time (context: where M sits in the cycle)
# --------------------------------------------------------------------------- #
def plot_phase_composition(df, out_prefix=None, show=False):
    plt = _new_axes(show)
    tcol = _time_col(df)
    phases = _phases_present(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, is_vme, title in (
        (axes[0], True, "VME phase composition"),
        (axes[1], False, "Control phase composition"),
    ):
        grp = df[df["is_vme"] == is_vme]
        if grp.empty:
            ax.set_title(f"{title}\n(no cells)")
            continue
        counts = grp.groupby([tcol, "phase"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=[p for p in phases if p in counts.columns], fill_value=0)
        frac = counts.div(counts.sum(axis=1).clip(lower=1), axis=0)
        ax.stackplot(
            frac.index,
            *[frac[p].values for p in frac.columns],
            labels=list(frac.columns),
            colors=[PHASE_COLORS.get(p, "#cccccc") for p in frac.columns],
            alpha=0.9,
        )
        if tcol == "frame_since_infection":
            ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_xlabel(tcol.replace("_", " "))
        ax.set_ylim(0, 1)
        ax.set_title(title)
    axes[0].set_ylabel("Fraction of cells")
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    _save(fig, plt, out_prefix, "phase_composition", show)
    return fig


# --------------------------------------------------------------------------- #
# 3. Fate of mitotic cells: divide vs. arrest vs. exit
# --------------------------------------------------------------------------- #
def mitotic_fate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Classify the fate of every cell (track) that is ever in M.

    Uses the lineage columns: a track that appears as a ``parent_track_id`` of
    another track divided. Otherwise we look at how its track ends.
    """
    tcol_end = "frame"
    last_frame = int(df[tcol_end].max())
    parents = set()
    if "parent_track_id" in df.columns:
        parents = set(int(p) for p in df["parent_track_id"].unique() if int(p) >= 0)

    ever_m = df[df["phase"].eq(MITOTIC_PHASE)]
    rows = []
    for tid, sub in df[df["track_id"].isin(ever_m["track_id"].unique())].groupby("track_id"):
        sub = sub.sort_values("frame")
        is_vme = bool(sub["is_vme"].any())
        track_end = int(sub["frame"].max())
        last_phase = sub.iloc[-1]["phase"]
        if int(tid) in parents:
            fate = "divided"
        elif track_end >= last_frame:
            fate = "censored (movie end)"
        elif last_phase == MITOTIC_PHASE:
            fate = "ended in M (arrest/loss)"
        else:
            fate = "exited to interphase"
        rows.append({"track_id": tid, "is_vme": is_vme, "fate": fate})
    return pd.DataFrame(rows)


def plot_mitotic_fate(df, out_prefix=None, show=False):
    plt = _new_axes(show)
    fate = mitotic_fate_table(df)
    if fate.empty:
        logger.warning("No mitotic cells found; skipping fate plot.")
        return None

    fate_order = ["divided", "exited to interphase", "ended in M (arrest/loss)", "censored (movie end)"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38
    groups = [(True, "VME", "#d000d0"), (False, "Control", "#555555")]
    x = np.arange(len(fate_order))
    for i, (is_vme, label, color) in enumerate(groups):
        sub = fate[fate["is_vme"] == is_vme]
        total = max(len(sub), 1)
        counts = [(sub["fate"] == f).sum() for f in fate_order]
        frac = [c / total for c in counts]
        bars = ax.bar(x + (i - 0.5) * width, frac, width, label=f"{label} (n={len(sub)})", color=color, alpha=0.85)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, str(c),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(fate_order, rotation=20, ha="right")
    ax.set_ylabel("Fraction of mitotic cells")
    ax.set_title("Fate of mitotic cells (VME vs control)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, plt, out_prefix, "mitotic_fate", show)
    return fig


# --------------------------------------------------------------------------- #
# 4. Phase that immediately follows M (one path or many?)
# --------------------------------------------------------------------------- #
def plot_post_mitotic_transitions(df, out_prefix=None, show=False):
    plt = _new_axes(show)
    rows = []
    for tid, sub in df.groupby("track_id"):
        sub = sub.sort_values("frame").reset_index(drop=True)
        ph = sub["phase"].tolist()
        is_vme = bool(sub["is_vme"].any())
        for i in range(len(ph) - 1):
            if ph[i] == MITOTIC_PHASE and ph[i + 1] != MITOTIC_PHASE:
                rows.append({"is_vme": is_vme, "next": ph[i + 1]})
    trans = pd.DataFrame(rows)
    if trans.empty:
        logger.warning("No post-mitotic transitions found; skipping transition plot.")
        return None

    nexts = [p for p in PHASE_ORDER if p in set(trans["next"].unique())]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.38
    x = np.arange(len(nexts))
    for i, (is_vme, label, color) in enumerate(((True, "VME", "#d000d0"), (False, "Control", "#555555"))):
        sub = trans[trans["is_vme"] == is_vme]
        total = max(len(sub), 1)
        frac = [(sub["next"] == p).sum() / total for p in nexts]
        ax.bar(x + (i - 0.5) * width, frac, width, label=f"{label} (n={len(sub)})",
               color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"M \u2192 {p}" for p in nexts], rotation=20, ha="right")
    ax.set_ylabel("Fraction of M exits")
    ax.set_title("Where mitotic cells go after M")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, plt, out_prefix, "post_mitotic_transitions", show)
    return fig


# --------------------------------------------------------------------------- #
# 5. Per-cell phase ribbons for mitotic VME cells (do they converge/diverge?)
# --------------------------------------------------------------------------- #
def plot_mitotic_trajectories(df, out_prefix=None, show=False, group_vme=True, max_cells=60):
    plt = _new_axes(show)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    tcol = _time_col(df)
    ever_m = df[df["phase"].eq(MITOTIC_PHASE) & (df["is_vme"] == group_vme)]["track_id"].unique()
    if len(ever_m) == 0:
        logger.warning("No mitotic %s cells; skipping trajectory ribbons.",
                       "VME" if group_vme else "control")
        return None

    sub = df[df["track_id"].isin(ever_m)].copy()
    phases = _phases_present(df)
    code = {p: i for i, p in enumerate(phases)}
    sub["pcode"] = sub["phase"].map(code)

    mat = sub.pivot_table(index="track_id", columns=tcol, values="pcode", aggfunc="first")
    # order rows by when each cell first becomes mitotic
    first_m = (
        sub[sub["phase"].eq(MITOTIC_PHASE)].groupby("track_id")[tcol].min().sort_values()
    )
    mat = mat.reindex(first_m.index)
    if len(mat) > max_cells:
        mat = mat.iloc[:max_cells]
        logger.info("Showing first %d of %d mitotic cells.", max_cells, len(first_m))

    cmap = ListedColormap([PHASE_COLORS.get(p, "#cccccc") for p in phases])
    norm = BoundaryNorm(np.arange(-0.5, len(phases) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(11, max(3, 0.18 * len(mat) + 1)))
    ax.imshow(
        mat.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
        extent=[mat.columns.min(), mat.columns.max(), len(mat), 0],
    )
    if tcol == "frame_since_infection":
        ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel(tcol.replace("_", " "))
    ax.set_ylabel("mitotic cells (1 row = 1 cell)")
    ax.set_yticks([])
    ax.set_title(f"Phase trajectories of mitotic {'VME' if group_vme else 'control'} cells")
    ax.legend(
        handles=[Patch(facecolor=PHASE_COLORS.get(p, "#ccc"), label=p) for p in phases],
        frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
    )
    fig.tight_layout()
    _save(fig, plt, out_prefix, f"mitotic_trajectories_{'vme' if group_vme else 'control'}", show)
    return fig


# --------------------------------------------------------------------------- #
# 6. Spatial snapshot of the microenvironment
# --------------------------------------------------------------------------- #
def plot_spatial_snapshot(df, out_prefix=None, show=False, frame: Optional[int] = None):
    plt = _new_axes(show)
    if not {"POSITION_X", "POSITION_Y"}.issubset(df.columns):
        return None
    if frame is None:
        # pick the frame with the most VME cells (the richest microenvironment)
        if df["is_vme"].any():
            frame = int(df[df["is_vme"]].groupby("frame").size().idxmax())
        else:
            frame = int(df["frame"].median())
    snap = df[df["frame"] == frame]

    fig, ax = plt.subplots(figsize=(7, 7))
    control = snap[~snap["is_vme"] & ~snap.get("is_infected", False)]
    ax.scatter(control["POSITION_X"], control["POSITION_Y"], s=14, c="#cccccc", label="control")
    if "is_infected" in snap.columns:
        inf = snap[snap["is_infected"]]
        ax.scatter(inf["POSITION_X"], inf["POSITION_Y"], s=60, c="#2ca02c",
                   marker="*", edgecolor="k", label="infected")
    vme = snap[snap["is_vme"]]
    is_m = vme["phase"].eq(MITOTIC_PHASE)
    ax.scatter(vme.loc[~is_m, "POSITION_X"], vme.loc[~is_m, "POSITION_Y"],
               s=30, c="#1f77b4", label="VME (non-M)")
    ax.scatter(vme.loc[is_m, "POSITION_X"], vme.loc[is_m, "POSITION_Y"],
               s=45, c="#d000d0", edgecolor="k", label="VME mitotic")
    ax.set_title(f"Microenvironment map @ frame {frame}")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, plt, out_prefix, "spatial_snapshot", show)
    return fig


# --------------------------------------------------------------------------- #
# Text summary
# --------------------------------------------------------------------------- #
def print_summary(df: pd.DataFrame) -> None:
    n_cells = df["cell_id"].nunique() if "cell_id" in df.columns else df["track_id"].nunique()
    print(f"Cells (tracks): {n_cells} | rows: {len(df)} | frames: {df['frame'].nunique()}")
    if "is_infected" in df.columns:
        print(f"Infected observations: {int(df['is_infected'].sum())}")
    if "is_vme" in df.columns:
        print(f"VME observations: {int(df['is_vme'].sum())} "
              f"| unique VME cells: {df.loc[df['is_vme'], 'cell_id'].nunique()}")
    fate = mitotic_fate_table(df)
    if not fate.empty:
        print("\nMitotic-cell fate (counts):")
        print(fate.groupby(["is_vme", "fate"]).size().unstack(fill_value=0))


def _build_parser():
    p = argparse.ArgumentParser(description="Visualize FUCCI-4 / VME results CSV.")
    p.add_argument("--csv", required=True, help="Path to pipeline results CSV.")
    p.add_argument("--out-prefix", default="viz", help="Prefix for saved PNGs.")
    p.add_argument("--show", action="store_true", help="Show figures interactively.")
    p.add_argument("--frame", type=int, default=None, help="Frame for the spatial snapshot.")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")
    df = load_results(args.csv)
    print_summary(df)

    out = None if args.show and args.out_prefix == "viz" else args.out_prefix
    plot_mitotic_fraction(df, out, args.show)
    plot_phase_composition(df, out, args.show)
    plot_mitotic_fate(df, out, args.show)
    plot_post_mitotic_transitions(df, out, args.show)
    plot_mitotic_trajectories(df, out, args.show, group_vme=True)
    plot_mitotic_trajectories(df, out, args.show, group_vme=False)
    plot_spatial_snapshot(df, out, args.show, frame=args.frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
