"""Morphology-based infection detection for the mNeonGreen-H1.0 FUCCI-4 line.

In this version of the FUCCI-4 system the constitutive DNA/nuclear marker is
**mNeonGreen-H1.0** (green channel) -- there is *no* separate GFP channel.
Infection is GFP-HCMV expressed as a **cell-surface** protein, so an infected
cell's green signal covers the whole cell and is large and irregular, whereas an
uninfected FUCCI-4 nucleus shows a small, round mNeonGreen-H1.0 dot.

Infection therefore cannot be called by intensity threshold (the nuclei are
green too). Instead we segment the green channel into objects and classify them
by **shape**: large + non-round (low circularity / solidity) = infected cell;
small + round = healthy nucleus. Infected pixels are then mapped back onto the
tracked nuclei (a nucleus whose centroid falls inside an infected green object
is flagged ``is_infected``), and that boolean feeds straight into the VME logic.

Implemented with numpy + scipy (``ndimage`` for connected components,
``spatial.ConvexHull`` for solidity); no scikit-image required.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull

try:
    from scipy.spatial import QhullError
except ImportError:  # older scipy
    from scipy.spatial.qhull import QhullError

from phase_fucci4 import estimate_threshold
from tracking import regionprops_numpy

logger = logging.getLogger("fucci_vme.morphology")


# --------------------------------------------------------------------------- #
# Extended region properties (shape descriptors)
# --------------------------------------------------------------------------- #
def extended_regionprops(
    labels: np.ndarray, compute_solidity: bool = True
) -> pd.DataFrame:
    """Per-label shape descriptors for one label image.

    Extends :func:`tracking.regionprops_numpy` with ``perimeter``, ``circularity``
    (``4*pi*area / perimeter**2``, in [0, 1]; ~1 for a disk, low for irregular),
    ``extent`` (area / bbox area) and ``solidity`` (area / convex-hull area).
    """
    base = regionprops_numpy(labels)
    cols = ["perimeter", "circularity", "extent", "solidity"]
    if base.empty:
        for c in cols:
            base[c] = pd.Series(dtype=float)
        return base

    n = int(labels.max())

    # perimeter via 4-connected boundary pixel count (pad to avoid wraparound)
    pad = np.pad(labels, 1)
    center = pad[1:-1, 1:-1]
    up, down = pad[:-2, 1:-1], pad[2:, 1:-1]
    left, right = pad[1:-1, :-2], pad[1:-1, 2:]
    boundary = (center > 0) & (
        (center != up) | (center != down) | (center != left) | (center != right)
    )
    perim = np.bincount(center[boundary], minlength=n + 1).astype(float)

    lab_idx = base["label"].to_numpy()
    base["perimeter"] = perim[lab_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        circ = 4.0 * math.pi * base["AREA"].to_numpy() / np.square(perim[lab_idx])
    base["circularity"] = np.clip(circ, 0.0, 1.0)

    if compute_solidity:
        extent = np.full(n + 1, np.nan)
        solidity = np.full(n + 1, np.nan)
        slices = ndimage.find_objects(labels)
        for i, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            sub = labels[sl] == i
            a = float(sub.sum())
            extent[i] = a / sub.size if sub.size else np.nan
            coords = np.argwhere(sub)
            if len(coords) >= 3:
                try:
                    hull = ConvexHull(coords.astype(float))
                    solidity[i] = a / hull.volume if hull.volume > 0 else np.nan
                except (QhullError, ValueError):
                    solidity[i] = np.nan
        base["extent"] = extent[lab_idx]
        base["solidity"] = solidity[lab_idx]
    else:
        base["extent"] = np.nan
        base["solidity"] = np.nan
    return base


# --------------------------------------------------------------------------- #
# Morphology-based infection detection
# --------------------------------------------------------------------------- #
def detect_infection_morphology(
    green: np.ndarray,
    masks: np.ndarray,
    spots: pd.DataFrame,
    area_factor: float = 2.5,
    min_circularity: float = 0.6,
    min_solidity: float = 0.85,
    use_solidity: bool = True,
    threshold_method: str = "otsu",
    abs_min_area: Optional[float] = None,
    assign: str = "overlap",
    min_overlap_frac: float = 0.5,
    infection_col: str = "is_infected",
) -> tuple[pd.DataFrame, dict]:
    """Flag tracked nuclei that sit inside a non-round (infected) green object.

    For each frame: threshold the green channel, find connected green objects,
    and     classify an object as *infected* when it is **large** (area greater than
    ``area_factor`` x the median object area, a robust proxy for nuclear size, or
    ``abs_min_area`` if given) **and** non-round (``circularity < min_circularity``
    or, when ``use_solidity``, ``solidity < min_solidity``).

    A tracked nucleus is then flagged ``is_infected`` by one of:
      * ``assign="overlap"`` (default): at least ``min_overlap_frac`` of the
        nucleus mask overlaps infected green pixels. More robust -- the cell whose
        nucleus is engulfed is flagged, not neighbors merely clipped at the edge.
      * ``assign="centroid"``: the nucleus centroid falls inside an infected
        object.

    Parameters mirror the qualitative criterion "the infected cell's surface GFP
    is large and not as round as a nuclear mNeonGreen-H1.0 dot" and are all
    tunable. Returns ``(spots_with_is_infected, info)``.
    """
    green = np.asarray(green)
    masks = np.asarray(masks)
    out = spots.copy()
    out[infection_col] = False

    frame_col = "frame"
    x_col, y_col = "POSITION_X", "POSITION_Y"
    H, W = green.shape[-2], green.shape[-1]

    per_frame_objs: list[dict] = []
    n_infected_objs = 0

    for t, fdf in out.groupby(frame_col):
        g = green[int(t)]
        thr = estimate_threshold(g.ravel(), method=threshold_method, log=True)
        green_bin = g > thr
        glab, n_obj = ndimage.label(green_bin)
        if n_obj == 0:
            continue
        props = extended_regionprops(glab, compute_solidity=use_solidity)
        if props.empty:
            continue

        med_area = float(np.median(props["AREA"]))
        area_gate = abs_min_area if abs_min_area is not None else area_factor * med_area
        large = props["AREA"].to_numpy() > area_gate
        irregular = props["circularity"].to_numpy() < min_circularity
        if use_solidity:
            sol = props["solidity"].to_numpy()
            irregular = irregular | (np.nan_to_num(sol, nan=1.0) < min_solidity)
        infected_obj = large & irregular

        infected_labels = props["label"].to_numpy()[infected_obj]
        n_infected_objs += int(infected_obj.sum())
        per_frame_objs.append(
            {"frame": int(t), "n_objects": int(n_obj),
             "median_area": med_area, "n_infected_objects": int(infected_obj.sum())}
        )
        if infected_labels.size == 0:
            continue

        infected_pixels = np.isin(glab, infected_labels)
        rows = fdf

        if assign == "overlap" and "label" in rows.columns:
            # fraction of each nucleus mask that overlaps infected green pixels
            mlab = masks[int(t)]
            n_nuc = int(mlab.max())
            inf_f = infected_pixels.astype(np.float64).ravel()
            denom = np.bincount(mlab.ravel(), minlength=n_nuc + 1).astype(np.float64)
            numer = np.bincount(mlab.ravel(), weights=inf_f, minlength=n_nuc + 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(denom > 0, numer / denom, 0.0)
            nuc_labels = rows["label"].to_numpy().astype(int)
            valid = nuc_labels < len(frac)
            calls = np.zeros(len(nuc_labels), dtype=bool)
            calls[valid] = frac[nuc_labels[valid]] >= min_overlap_frac
            out.loc[rows.index, infection_col] = calls
        else:
            cy = np.clip(np.round(rows[y_col].to_numpy()).astype(int), 0, H - 1)
            cx = np.clip(np.round(rows[x_col].to_numpy()).astype(int), 0, W - 1)
            out.loc[rows.index, infection_col] = infected_pixels[cy, cx]

    info = {
        "n_infected_observations": int(out[infection_col].sum()),
        "n_infected_objects_total": n_infected_objs,
        "params": {
            "area_factor": area_factor,
            "abs_min_area": abs_min_area,
            "min_circularity": min_circularity,
            "min_solidity": min_solidity if use_solidity else None,
            "threshold_method": threshold_method,
        },
        "per_frame": pd.DataFrame(per_frame_objs),
    }
    logger.info(
        "Morphology infection: %d infected nucleus-observations across frames "
        "(%d infected green objects).",
        info["n_infected_observations"], n_infected_objs,
    )
    return out, info
