"""Virus Microenvironment (VME) analysis for ConfluentFUCCI tracking output.

This standalone module ingests the tracked dataframe produced by ConfluentFUCCI
(https://github.com/leogolds/ConfluentFUCCI) and applies spatial logic to define
"Virus Microenvironments" (VMEs) around GFP-infected cells, then visualizes the
cell-cycle trajectories of the neighboring (VME) cells versus the distal control
population.

A VME is the set of uninfected cells that are immediately adjacent (share a
Delaunay edge) to an infected cell at a given moment in time. Because infection
can involve several adjacent cells, contiguous infected cells are grouped into a
single VME via connected components of the Delaunay graph.

Expected input
--------------
The primary input is ConfluentFUCCI's ``confluent_fucci_data.csv`` (the output of
``CartesianSimilarity.get_all_spots``). The columns this module relies on are:

============== =====================================================
column         meaning
============== =====================================================
frame          integer time index of the movie
POSITION_X     nuclear centroid x coordinate (pixels)
POSITION_Y     nuclear centroid y coordinate (pixels)
color          FUCCI phase proxy: 'red'=G1, 'yellow'=G1/S, 'green'=S/G2/M
merged_track_id cell identity for tracked (merged red+green) cells, e.g. 'r12_g34'
source_track   'red' / 'green' provenance of an unmerged spot
track_id       per-channel track id (when available)
ID             unique spot id for a single detection in a single frame
============== =====================================================

The trimmed CSV does **not** retain a GFP channel intensity. Provide the GFP
column via ``gfp_col`` (or merge it in yourself before calling this module). The
module will also auto-detect common TrackMate-style names such as
``MEAN_INTENSITY_CH3`` / ``mean_intensity_gfp``.

Column resolution is alias-driven (see ``COLUMN_ALIASES``) so raw TrackMate
exports (``FRAME``, ``TRACK_ID`` ...) also work.

Typical usage
-------------
    python fucci_vme.py --input confluent_fucci_data.csv \
        --gfp-col MEAN_INTENSITY_CH3 --gfp-threshold 500 \
        --out-csv vme_tagged.csv --out-fig-prefix vme

    # or from Python
    from fucci_vme import VMEConfig, run_vme_analysis
    cfg = VMEConfig(gfp_col="MEAN_INTENSITY_CH3", gfp_threshold=500.0)
    tagged, index_cells = run_vme_analysis(df, cfg)
"""

from __future__ import annotations

import argparse
import itertools
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

try:  # scipy >= 1.8 exposes QhullError at the package root
    from scipy.spatial import QhullError
except ImportError:  # older scipy
    from scipy.spatial.qhull import QhullError

logger = logging.getLogger("fucci_vme")


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #
COLUMN_ALIASES: dict[str, Sequence[str]] = {
    "frame": ("frame", "FRAME", "Frame", "t", "T", "POSITION_T"),
    "x": ("POSITION_X", "x", "X", "center_x", "centroid_x"),
    "y": ("POSITION_Y", "y", "Y", "center_y", "centroid_y"),
    "phase": ("color", "phase", "PHASE", "cell_cycle_phase"),
    "gfp": (
        "mean_intensity_gfp",
        "gfp",
        "GFP",
        "mean_gfp",
        "MEAN_GFP",
        "MEAN_INTENSITY_CH3",
        "MEAN_INTENSITY_CH4",
        "MEAN_INTENSITY_GFP",
    ),
}

# FUCCI phase proxy used by ConfluentFUCCI. Ordered along the cell cycle so the
# phase can be rendered as an ordinal axis (G1 -> S -> G2/M).
DEFAULT_PHASE_ORDER: tuple[str, ...] = ("red", "yellow", "green")
DEFAULT_PHASE_LABELS: dict[str, str] = {
    "red": "G1/G0",
    "yellow": "G1/S",
    "green": "S/G2/M",
}


def _resolve_column(
    df: pd.DataFrame, role: str, override: Optional[str] = None
) -> Optional[str]:
    """Return the actual dataframe column for a logical ``role``.

    ``override`` wins if provided. Otherwise the first matching alias is used.
    Returns ``None`` if nothing matches (callers decide whether that is fatal).
    """
    if override is not None:
        if override not in df.columns:
            raise KeyError(
                f"Requested {role} column {override!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return override
    for candidate in COLUMN_ALIASES.get(role, ()):
        if candidate in df.columns:
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class VMEConfig:
    """Configuration for a VME analysis run."""

    # --- column overrides (None -> auto-detect via COLUMN_ALIASES) ---
    frame_col: Optional[str] = None
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    phase_col: Optional[str] = None
    gfp_col: Optional[str] = None
    id_col: Optional[str] = None  # cell identity column; auto-built if None
    # Precomputed boolean infection column. When set, infection is read directly
    # from this column (per frame) instead of thresholding GFP intensity. This is
    # how the morphology-based pipeline (cell-surface GFP / mNeonGreen-H1.0
    # version) feeds its shape-based infection calls into the VME logic.
    infection_col: Optional[str] = None

    # --- infection / VME logic ---
    gfp_threshold: float = 0.0
    # If True, the infected status is evaluated per-frame (dynamic). The "index"
    # cell is still the first cell to cross threshold.
    dynamic_infection: bool = True
    # Only build a VME around contiguous infected components that contain at
    # least one index cell. Set False to treat every infected cell as a seed.
    require_index: bool = True
    # Cells closer than this many pixels are treated as duplicates and one is
    # dropped before triangulation (avoids QhullError on coincident points).
    min_point_separation: float = 1e-6

    # --- phase encoding for plotting ---
    phase_order: Sequence[str] = field(default_factory=lambda: DEFAULT_PHASE_ORDER)
    phase_labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_PHASE_LABELS)
    )


# --------------------------------------------------------------------------- #
# Loading & identity
# --------------------------------------------------------------------------- #
def load_tracked_data(path: str | Path) -> pd.DataFrame:
    """Load a ConfluentFUCCI tracked dataframe from CSV / Parquet / HDF5."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".h5", ".hdf5", ".hdf"}:
        df = pd.read_hdf(path)
    else:
        raise ValueError(f"Unsupported file type {suffix!r} for {path}")
    logger.info("Loaded %d rows x %d cols from %s", len(df), df.shape[1], path)
    return df


def add_cell_id(df: pd.DataFrame, cfg: VMEConfig) -> tuple[pd.DataFrame, str]:
    """Ensure a stable per-cell identity column exists.

    Preference order:
      1. explicit ``cfg.id_col``
      2. ``merged_track_id`` (skipping the literal 'unmerged' sentinel)
      3. ``source_track`` + ``track_id`` / ``TRACK_ID``
      4. fall back to the per-detection ``ID`` / ``id`` (no cross-frame linkage)

    Returns the (possibly augmented) dataframe and the name of the id column.
    """
    df = df.copy()

    if cfg.id_col is not None:
        if cfg.id_col not in df.columns:
            raise KeyError(f"id_col {cfg.id_col!r} not in dataframe")
        return df, cfg.id_col

    if "cell_id" in df.columns:
        return df, "cell_id"

    def build_id(row: pd.Series) -> str:
        mt = row.get("merged_track_id")
        if isinstance(mt, str) and mt and mt != "unmerged":
            return mt
        src = row.get("source_track")
        tid = row.get("track_id", row.get("TRACK_ID"))
        if pd.notna(tid):
            prefix = str(src) if pd.notna(src) else "trk"
            return f"{prefix}_{int(tid)}" if float(tid).is_integer() else f"{prefix}_{tid}"
        sid = row.get("ID", row.get("id"))
        if pd.notna(sid):
            return f"spot_{int(sid)}" if float(sid).is_integer() else f"spot_{sid}"
        return f"row_{row.name}"

    df["cell_id"] = df.apply(build_id, axis=1)

    n_unlinked = df["cell_id"].str.startswith("spot_").sum()
    if n_unlinked:
        logger.warning(
            "%d/%d detections have no cross-frame track link (unmerged spots). "
            "They contribute to Delaunay topology but cannot form multi-frame "
            "trajectories.",
            n_unlinked,
            len(df),
        )
    return df, "cell_id"


# --------------------------------------------------------------------------- #
# Step 1: Identify the index infection(s)
# --------------------------------------------------------------------------- #
def find_index_infections(
    df: pd.DataFrame,
    cfg: VMEConfig,
    id_col: str,
) -> pd.DataFrame:
    """Find, per cell, the first frame it is infected.

    Infection is read from ``cfg.infection_col`` (a precomputed boolean column,
    e.g. from morphology-based detection) when provided; otherwise from the first
    frame the GFP intensity exceeds ``cfg.gfp_threshold``.

    Returns a dataframe with columns ``[id_col, 'infection_frame', 'gfp_at_infection']``
    sorted by infection frame. Each row is an "Index Cell".
    """
    frame_col = _resolve_column(df, "frame", cfg.frame_col)
    if frame_col is None:
        raise KeyError("Could not resolve a 'frame' column.")

    # Decide the infection criterion: precomputed boolean column vs GFP threshold.
    if cfg.infection_col is not None:
        if cfg.infection_col not in df.columns:
            raise KeyError(f"infection_col {cfg.infection_col!r} not in dataframe.")
        infected = df[df[cfg.infection_col].astype(bool)]
        value_col = cfg.infection_col
        criterion = f"column {cfg.infection_col!r} is True"
    else:
        gfp_col = _resolve_column(df, "gfp", cfg.gfp_col)
        if gfp_col is None:
            raise KeyError(
                "Could not resolve a GFP intensity column. Pass cfg.gfp_col or "
                f"cfg.infection_col explicitly (looked for {COLUMN_ALIASES['gfp']})."
            )
        infected = df[df[gfp_col] > cfg.gfp_threshold]
        value_col = gfp_col
        criterion = f"GFP > {cfg.gfp_threshold:.3g} in {gfp_col!r}"

    if infected.empty:
        logger.warning("No cell is infected (%s).", criterion)
        return pd.DataFrame(columns=[id_col, "infection_frame", "gfp_at_infection"])

    first_cross = (
        infected.sort_values(frame_col)
        .groupby(id_col, as_index=False)
        .first()
    )
    out = first_cross[[id_col, frame_col, value_col]].rename(
        columns={frame_col: "infection_frame", value_col: "gfp_at_infection"}
    )
    out = out.sort_values("infection_frame").reset_index(drop=True)
    logger.info("Identified %d index (infected) cell(s).", len(out))
    return out


# --------------------------------------------------------------------------- #
# Step 2: Spatial connectivity via Delaunay
# --------------------------------------------------------------------------- #
def _dedupe_points(
    ids: np.ndarray, xy: np.ndarray, min_sep: float
) -> tuple[np.ndarray, np.ndarray]:
    """Drop coincident/near-coincident points that would break Qhull."""
    if min_sep <= 0 or len(xy) < 2:
        return ids, xy
    keep = np.ones(len(xy), dtype=bool)
    rounded = np.round(xy / max(min_sep, 1e-12)).astype(np.int64)
    seen: set[tuple[int, int]] = set()
    for i, key in enumerate(map(tuple, rounded)):
        if key in seen:
            keep[i] = False
        else:
            seen.add(key)
    return ids[keep], xy[keep]


def build_delaunay_adjacency(
    frame_df: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    min_sep: float = 1e-6,
) -> dict[object, set]:
    """Compute the Delaunay graph for one frame.

    Returns a mapping ``cell_id -> set(neighbor cell_ids)``. Two cells are
    neighbors iff they share a Delaunay edge (the 1-ring / natural neighbors).
    Returns an empty adjacency when fewer than 3 non-degenerate points exist.
    """
    sub = frame_df[[id_col, x_col, y_col]].dropna(subset=[x_col, y_col])
    ids = sub[id_col].to_numpy()
    xy = sub[[x_col, y_col]].to_numpy(dtype=float)

    ids, xy = _dedupe_points(ids, xy, min_sep)

    adjacency: dict[object, set] = {cid: set() for cid in ids}
    if len(xy) < 3:
        return adjacency

    try:
        tri = Delaunay(xy)
    except QhullError as exc:  # collinear / degenerate point cloud
        logger.debug("Delaunay failed for a frame (%s); no edges.", exc)
        return adjacency

    for simplex in tri.simplices:
        for a, b in itertools.combinations(simplex, 2):
            ca, cb = ids[a], ids[b]
            adjacency[ca].add(cb)
            adjacency[cb].add(ca)
    return adjacency


def _connected_components(
    nodes: Iterable, adjacency: dict[object, set]
) -> list[set]:
    """Connected components restricted to ``nodes`` using ``adjacency`` edges."""
    node_set = set(nodes)
    seen: set = set()
    components: list[set] = []
    for start in node_set:
        if start in seen:
            continue
        comp: set = set()
        queue = deque([start])
        seen.add(start)
        while queue:
            cur = queue.popleft()
            comp.add(cur)
            for nb in adjacency.get(cur, ()):  # only walk within node_set
                if nb in node_set and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


# --------------------------------------------------------------------------- #
# Step 3: Define VME neighbors (with contiguity)
# --------------------------------------------------------------------------- #
def tag_vme(
    df: pd.DataFrame,
    cfg: VMEConfig,
    id_col: Optional[str] = None,
    index_cells: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tag every row with infection / index / VME membership per frame.

    For each frame at or after the earliest infection:
      * mark infected cells (GFP > threshold; static or dynamic per cfg),
      * group contiguous infected cells into VME seeds (Delaunay components),
      * mark uninfected cells sharing a Delaunay edge with a seed as ``is_vme``.

    Adds the columns:
      ``is_infected``  - GFP above threshold at that frame
      ``is_index``     - first-crossing (index) cell, at/after its infection frame
      ``is_vme``       - uninfected neighbor of an infected (index-containing) seed
      ``vme_id``       - per-frame id of the contiguous VME the cell borders
      ``vme_index_id`` - stable id of the VME = sorted index cells in the seed
      ``frame_since_infection`` - frame minus the global earliest infection frame

    Returns ``(tagged_df, index_cells)``.
    """
    frame_col = _resolve_column(df, "frame", cfg.frame_col)
    x_col = _resolve_column(df, "x", cfg.x_col)
    y_col = _resolve_column(df, "y", cfg.y_col)
    gfp_col = None if cfg.infection_col is not None else _resolve_column(df, "gfp", cfg.gfp_col)
    for role, col in (("frame", frame_col), ("x", x_col), ("y", y_col)):
        if col is None:
            raise KeyError(f"Could not resolve a {role!r} column.")

    if id_col is None:
        df, id_col = add_cell_id(df, cfg)

    if index_cells is None:
        index_cells = find_index_infections(df, cfg, id_col)

    df = df.copy()
    df["is_infected"] = False
    df["is_index"] = False
    df["is_vme"] = False
    df["vme_id"] = pd.Series([pd.NA] * len(df), dtype="object")
    df["vme_index_id"] = pd.Series([pd.NA] * len(df), dtype="object")

    if index_cells.empty:
        df["frame_since_infection"] = pd.NA
        return df, index_cells

    # Map cell_id -> its index infection frame (for is_index and static mode).
    infection_frame_of = dict(
        zip(index_cells[id_col], index_cells["infection_frame"])
    )
    index_id_set = set(infection_frame_of)
    earliest = int(index_cells["infection_frame"].min())
    df["frame_since_infection"] = df[frame_col] - earliest

    # Work on positional row indices for fast assignment.
    df = df.reset_index(drop=True)

    for frame, frame_df in df.groupby(frame_col, sort=True):
        if frame < earliest:
            continue

        # --- determine infected cells in this frame ---
        if cfg.infection_col is not None:
            infected_mask = frame_df[cfg.infection_col].astype(bool)
        elif cfg.dynamic_infection and gfp_col is not None:
            infected_mask = frame_df[gfp_col] > cfg.gfp_threshold
        else:
            # static: a cell is infected from its index infection frame onward
            infected_mask = frame_df[id_col].map(
                lambda c: c in infection_frame_of and frame >= infection_frame_of[c]
            )
        infected_ids = set(frame_df.loc[infected_mask, id_col])

        # index cells that have already been infected by this frame
        active_index_ids = {
            c for c in index_id_set if frame >= infection_frame_of[c]
        }
        # ensure active index cells count as infected seeds even if their GFP
        # momentarily dips below threshold
        infected_ids |= (active_index_ids & set(frame_df[id_col]))

        df.loc[frame_df.index[frame_df[id_col].isin(infected_ids)], "is_infected"] = True
        df.loc[frame_df.index[frame_df[id_col].isin(active_index_ids)], "is_index"] = True

        if not infected_ids:
            continue

        adjacency = build_delaunay_adjacency(
            frame_df, id_col, x_col, y_col, cfg.min_point_separation
        )
        if not adjacency:
            continue

        # contiguous infected components (each is one VME seed)
        components = _connected_components(
            infected_ids & set(adjacency), adjacency
        )

        for comp_i, comp in enumerate(components):
            comp_index_ids = sorted(str(c) for c in (comp & index_id_set))
            if cfg.require_index and not comp_index_ids:
                continue
            vme_label = f"f{int(frame)}_c{comp_i}"
            stable_label = "+".join(comp_index_ids) if comp_index_ids else vme_label

            # uninfected 1-ring neighbors of the whole infected component
            neighbors: set = set()
            for inf in comp:
                neighbors |= adjacency.get(inf, set())
            neighbors -= infected_ids  # VME = uninfected only

            if not neighbors:
                continue
            sel = frame_df.index[frame_df[id_col].isin(neighbors)]
            df.loc[sel, "is_vme"] = True
            # only fill where not already assigned (first seed wins on ties)
            unset = sel[df.loc[sel, "vme_id"].isna()]
            df.loc[unset, "vme_id"] = vme_label
            df.loc[unset, "vme_index_id"] = stable_label

    n_vme_rows = int(df["is_vme"].sum())
    n_vme_cells = df.loc[df["is_vme"], id_col].nunique()
    logger.info(
        "Tagged %d VME observations spanning %d unique cell(s).",
        n_vme_rows,
        n_vme_cells,
    )
    return df, index_cells


# --------------------------------------------------------------------------- #
# Step 4: Phase trajectories
# --------------------------------------------------------------------------- #
def _phase_to_ordinal(
    series: pd.Series, phase_order: Sequence[str]
) -> pd.Series:
    mapping = {p: i for i, p in enumerate(phase_order)}
    return series.map(mapping)


def summarize_phase_fractions(
    df: pd.DataFrame,
    cfg: VMEConfig,
    group_col: str = "is_vme",
    time_col: str = "frame_since_infection",
) -> pd.DataFrame:
    """Fraction of cells in each phase per timepoint, split by ``group_col``.

    Returns a tidy dataframe: ``[time_col, group_col, phase, fraction, n]``.
    """
    phase_col = _resolve_column(df, "phase", cfg.phase_col)
    if phase_col is None:
        raise KeyError("Could not resolve a phase/color column.")
    tcol = time_col if time_col in df.columns else _resolve_column(df, "frame", cfg.frame_col)

    counts = (
        df.groupby([tcol, group_col, phase_col]).size().rename("n").reset_index()
    )
    totals = counts.groupby([tcol, group_col])["n"].transform("sum")
    counts["fraction"] = counts["n"] / totals
    counts = counts.rename(columns={phase_col: "phase"})
    return counts


def plot_phase_trajectories(
    df: pd.DataFrame,
    cfg: VMEConfig,
    out_prefix: Optional[str | Path] = None,
    time_col: str = "frame_since_infection",
    show: bool = False,
):
    """Plot VME vs control cell-cycle behavior.

    Produces three panels:
      1. mean phase (ordinal) over time +/- SEM, VME vs control
      2/3. stacked phase-fraction area plots over time for each group

    Returns the matplotlib Figure.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase_col = _resolve_column(df, "phase", cfg.phase_col)
    if phase_col is None:
        raise KeyError("Could not resolve a phase/color column.")
    tcol = time_col if time_col in df.columns else _resolve_column(df, "frame", cfg.frame_col)

    work = df.dropna(subset=[tcol]).copy()
    work["phase_ord"] = _phase_to_ordinal(work[phase_col], cfg.phase_order)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel 1: mean phase ordinal over time ---
    ax = axes[0]
    for is_vme, label, color in (
        (True, "VME (neighbors)", "#d62728"),
        (False, "Control (distal)", "#1f77b4"),
    ):
        grp = work[work["is_vme"] == is_vme]
        if grp.empty:
            continue
        stats = grp.groupby(tcol)["phase_ord"].agg(["mean", "sem"]).reset_index()
        ax.plot(stats[tcol], stats["mean"], label=label, color=color, lw=2)
        ax.fill_between(
            stats[tcol],
            stats["mean"] - stats["sem"].fillna(0),
            stats["mean"] + stats["sem"].fillna(0),
            color=color,
            alpha=0.2,
        )
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel(_axis_label(tcol))
    ax.set_yticks(range(len(cfg.phase_order)))
    ax.set_yticklabels([cfg.phase_labels.get(p, p) for p in cfg.phase_order])
    ax.set_ylabel("Mean cell-cycle phase")
    ax.set_title("Mean phase over time")
    ax.legend(frameon=False, fontsize=9)

    # --- Panels 2 & 3: stacked phase fractions ---
    fractions = summarize_phase_fractions(work, cfg, time_col=tcol)
    phase_colors = _phase_palette(cfg.phase_order)
    for ax, is_vme, title in (
        (axes[1], True, "VME phase composition"),
        (axes[2], False, "Control phase composition"),
    ):
        grp = fractions[fractions["is_vme"] == is_vme]
        if grp.empty:
            ax.set_title(f"{title}\n(no cells)")
            continue
        pivot = (
            grp.pivot_table(index=tcol, columns="phase", values="fraction", fill_value=0)
            .reindex(columns=[p for p in cfg.phase_order if p in grp["phase"].unique()])
        )
        ax.stackplot(
            pivot.index,
            *[pivot[p].values for p in pivot.columns],
            labels=[cfg.phase_labels.get(p, p) for p in pivot.columns],
            colors=[phase_colors[p] for p in pivot.columns],
            alpha=0.9,
        )
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_xlabel(_axis_label(tcol))
        ax.set_ylabel("Fraction of cells")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    if out_prefix is not None:
        path = f"{out_prefix}_phase_trajectories.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Wrote %s", path)
    if show:
        plt.show()
    return fig


def _axis_label(tcol: str) -> str:
    return "Frames since infection" if "since" in tcol else f"{tcol}"


def _phase_palette(phase_order: Sequence[str]) -> dict[str, str]:
    preset = {"red": "#d62728", "yellow": "#e7c000", "green": "#2ca02c"}
    fallback = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    palette: dict[str, str] = {}
    for i, p in enumerate(phase_order):
        palette[p] = preset.get(p, fallback[i % len(fallback)])
    return palette


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_vme_analysis(
    df: pd.DataFrame, cfg: VMEConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: identity -> index detection -> VME tagging.

    Returns ``(tagged_df, index_cells)``.
    """
    df, id_col = add_cell_id(df, cfg)
    index_cells = find_index_infections(df, cfg, id_col)
    tagged, index_cells = tag_vme(df, cfg, id_col=id_col, index_cells=index_cells)
    return tagged, index_cells


# --------------------------------------------------------------------------- #
# Synthetic data (for testing without the real experiment)
# --------------------------------------------------------------------------- #
def generate_synthetic_dataset(
    n_frames: int = 30,
    grid: int = 8,
    spacing: float = 30.0,
    jitter: float = 3.0,
    infection_frame: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a small synthetic ConfluentFUCCI-like dataframe for testing.

    A grid of cells is tracked over time. The central cell becomes GFP positive
    at ``infection_frame``; its immediate neighbors are biased toward an
    'arrested' (green / S-G2-M) phase afterwards to mimic VME dysregulation.
    """
    rng = np.random.default_rng(seed)
    phases = list(DEFAULT_PHASE_ORDER)
    coords = [(i, j) for i in range(grid) for j in range(grid)]
    center = (grid // 2, grid // 2)
    center_id = f"r{center[0] * grid + center[1]}_g{center[0] * grid + center[1]}"

    rows = []
    for (i, j) in coords:
        cell_idx = i * grid + j
        cell_id = f"r{cell_idx}_g{cell_idx}"
        base_x = i * spacing
        base_y = j * spacing
        is_center = (i, j) == center
        is_ring = abs(i - center[0]) <= 1 and abs(j - center[1]) <= 1 and not is_center
        phase_cursor = rng.integers(0, len(phases))
        for f in range(n_frames):
            phase_cursor = (phase_cursor + (rng.random() < 0.3)) % len(phases)
            if is_ring and f >= infection_frame and rng.random() < 0.7:
                phase = "green"  # mimic G2/M arrest in the VME
            else:
                phase = phases[phase_cursor]
            gfp = 0.0
            if is_center and f >= infection_frame:
                gfp = 800.0 + 50.0 * (f - infection_frame) + rng.normal(0, 20)
            else:
                gfp = rng.normal(50, 15)
            rows.append(
                {
                    "frame": f,
                    "POSITION_X": base_x + rng.normal(0, jitter),
                    "POSITION_Y": base_y + rng.normal(0, jitter),
                    "color": phase,
                    "merged_track_id": cell_id,
                    "source_track": "merged",
                    "track_id": cell_idx,
                    "ID": cell_idx * n_frames + f,
                    "MEAN_INTENSITY_CH3": max(gfp, 0.0),
                    "AREA": rng.normal(120, 10),
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["center_id"] = center_id
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Define Virus Microenvironments (VMEs) from ConfluentFUCCI "
        "tracking output and plot VME vs control cell-cycle trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=str, help="Path to ConfluentFUCCI CSV/Parquet/HDF.")
    p.add_argument("--demo", action="store_true", help="Run on a synthetic dataset.")
    p.add_argument("--gfp-col", type=str, default=None, help="GFP intensity column name.")
    p.add_argument("--gfp-threshold", type=float, default=0.0, help="Infection threshold.")
    p.add_argument("--frame-col", type=str, default=None)
    p.add_argument("--x-col", type=str, default=None)
    p.add_argument("--y-col", type=str, default=None)
    p.add_argument("--phase-col", type=str, default=None)
    p.add_argument("--id-col", type=str, default=None)
    p.add_argument(
        "--static-infection",
        action="store_true",
        help="Treat a cell as infected from its index frame onward (vs per-frame GFP).",
    )
    p.add_argument(
        "--no-require-index",
        action="store_true",
        help="Build VMEs around every infected component, not just index-containing ones.",
    )
    p.add_argument("--out-csv", type=str, default=None, help="Where to write the tagged dataframe.")
    p.add_argument("--out-fig-prefix", type=str, default=None, help="Prefix for output figures.")
    p.add_argument("--show", action="store_true", help="Display figures interactively.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.demo or not args.input:
        if not args.demo:
            logger.warning("No --input given; running --demo on synthetic data.")
        df = generate_synthetic_dataset()
        gfp_col = args.gfp_col or "MEAN_INTENSITY_CH3"
        gfp_threshold = args.gfp_threshold or 400.0
    else:
        df = load_tracked_data(args.input)
        gfp_col = args.gfp_col
        gfp_threshold = args.gfp_threshold

    cfg = VMEConfig(
        frame_col=args.frame_col,
        x_col=args.x_col,
        y_col=args.y_col,
        phase_col=args.phase_col,
        gfp_col=gfp_col,
        id_col=args.id_col,
        gfp_threshold=gfp_threshold,
        dynamic_infection=not args.static_infection,
        require_index=not args.no_require_index,
    )

    tagged, index_cells = run_vme_analysis(df, cfg)

    print("\n=== Index (infected) cells ===")
    if index_cells.empty:
        print("  none detected above threshold")
    else:
        print(index_cells.to_string(index=False))

    _, id_col = add_cell_id(tagged, cfg)
    vme_rows = tagged.loc[tagged["is_vme"]]
    print(f"\nVME observations: {len(vme_rows)}")
    print(f"Unique VME cells: {vme_rows[id_col].nunique()}")

    if args.out_csv:
        tagged.to_csv(args.out_csv, index=False)
        logger.info("Wrote tagged dataframe to %s", args.out_csv)

    if args.out_fig_prefix or args.show:
        plot_phase_trajectories(
            tagged, cfg, out_prefix=args.out_fig_prefix, show=args.show
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
