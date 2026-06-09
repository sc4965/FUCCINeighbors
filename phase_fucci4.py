"""Auto-gating cell-cycle phase classifier for the modified FUCCI-4 line.

The FUCCI-4 constructs (see project README) report the cycle with three
phase-specific markers plus a constitutive nuclear marker:

    mScarlet3 - Cdt1(30-120)    -> high in  G1/G0   (APC/C activity)
    emiRFP670 - Geminin(1-110)  -> high in  S/G2/M  (SCF activity)
    mTagBFP2  - SLBP(18-126)    -> high in  S       (replication-dependent)
    miRFP720  - H1.0            -> constitutive nucleus (segmentation/mitosis)

Because absolute intensities are experiment-specific, this module **auto-gates**
each marker into positive/negative populations (Otsu by default, optional
2-component Gaussian mixture, or fixed quantile) and then combines the boolean
calls into a phase via an editable truth table. The chosen thresholds are
returned so they can be inspected and overridden later for tuning.

Main entry point: :func:`classify_fucci4`.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("fucci_vme.phase")

# Logical marker -> the cell-cycle compartment it reports.
DEFAULT_MARKER_ROLES = ("cdt1", "geminin", "slbp")

# Truth table keyed by (cdt1+, geminin+, slbp+) booleans -> phase label.
# Editable: tweak any cell to change the semantics of a marker combination.
DEFAULT_TRUTH_TABLE: dict[tuple[bool, bool, bool], str] = {
    (False, False, False): "G0",      # all low: quiescent / undetermined
    (True, False, False): "G1",       # Cdt1 only
    (True, False, True): "G1/S",      # Cdt1 + SLBP (entering S)
    (False, False, True): "S",        # SLBP only
    (False, True, True): "S",         # SLBP + Geminin (active replication)
    (True, True, True): "S",          # ambiguous, all on -> call S
    (False, True, False): "G2/M",     # Geminin only
    (True, True, False): "G2/M",      # Geminin + Cdt1 (M->G1 reappearance)
}

# Order along the cycle for plotting / ordinal encoding.
DEFAULT_PHASE_ORDER = ("G0", "G1", "G1/S", "S", "G2/M", "M")
DEFAULT_PHASE_LABELS = {
    "G0": "G0", "G1": "G1", "G1/S": "G1/S", "S": "S", "G2/M": "G2/M", "M": "M",
}


# --------------------------------------------------------------------------- #
# Threshold estimation (auto-gating)
# --------------------------------------------------------------------------- #
def _otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu's threshold (pure numpy) on the given 1D values."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    hist, edges = np.histogram(v, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(np.float64)
    total = w.sum()
    if total == 0:
        return float(np.median(v))
    w = w / total
    cum_w = np.cumsum(w)
    cum_mean = np.cumsum(w * centers)
    global_mean = cum_mean[-1]
    denom = cum_w * (1.0 - cum_w)
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (global_mean * cum_w - cum_mean) ** 2 / denom
    between[~np.isfinite(between)] = -np.inf
    return float(centers[int(np.argmax(between))])


def _gmm_threshold(values: np.ndarray) -> float:
    """Crossover threshold from a 2-component 1D Gaussian mixture (sklearn)."""
    from sklearn.mixture import GaussianMixture

    v = values[np.isfinite(values)].reshape(-1, 1)
    if len(v) < 2:
        return float("nan")
    gm = GaussianMixture(n_components=2, random_state=0).fit(v)
    lo, hi = np.sort(gm.means_.ravel())
    grid = np.linspace(lo, hi, 512).reshape(-1, 1)
    pred = gm.predict(grid)
    hi_comp = int(np.argmax(gm.means_.ravel()))
    switch = np.where(pred == hi_comp)[0]
    return float(grid[switch[0], 0]) if switch.size else float((lo + hi) / 2)


def estimate_threshold(
    values: np.ndarray,
    method: str = "otsu",
    log: bool = True,
    quantile: float = 0.5,
) -> float:
    """Estimate a positive/negative gate for one marker.

    ``method`` is one of ``'otsu'``, ``'gmm'`` or ``'quantile'``. When ``log`` is
    True the gate is computed in ``log1p`` space and mapped back to raw units.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    x = np.log1p(np.clip(v, 0, None)) if log else v
    if method == "otsu":
        thr = _otsu_threshold(x)
    elif method == "gmm":
        thr = _gmm_threshold(x)
    elif method == "quantile":
        thr = float(np.quantile(x, quantile))
    else:
        raise ValueError(f"Unknown gating method {method!r}")
    return float(np.expm1(thr)) if log else float(thr)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classify_fucci4(
    df: pd.DataFrame,
    marker_cols: Mapping[str, str],
    method: str = "otsu",
    log: bool = True,
    per_frame: bool = False,
    thresholds: Optional[Mapping[str, float]] = None,
    truth_table: Optional[Mapping[tuple, str]] = None,
    detect_mitosis: bool = False,
    eccentricity_col: str = "eccentricity",
    mitosis_min_eccentricity: Optional[float] = 0.8,
    mitosis_quantile: float = 0.9,
    phase_col: str = "phase",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assign a FUCCI-4 cell-cycle phase to every row.

    Parameters
    ----------
    df:
        Spots dataframe with per-nucleus marker intensity columns.
    marker_cols:
        Mapping of logical marker -> column name, e.g.
        ``{"cdt1": "cdt1_mean", "geminin": "geminin_mean", "slbp": "slbp_mean"}``.
    method, log, per_frame:
        Auto-gating controls. ``per_frame=True`` computes a separate threshold
        for each frame (useful when intensity drifts with bleaching).
    thresholds:
        Optional explicit ``{marker: value}`` overrides (skips auto-gating for
        those markers). This is the hook for tuning after inspecting the
        returned auto thresholds.
    truth_table:
        Optional override of the (cdt1+, geminin+, slbp+) -> phase mapping.
    detect_mitosis:
        If True, reclassify ``G2/M`` cells whose H1.0 chromatin is **condensed**
        into ``M``. During mitosis the histone-marked chromatin forms an
        elongated, rectangular metaphase plate -> high ``eccentricity_col``. A
        cell in the Geminin-high (G2/M) window with eccentricity at or above
        ``mitosis_min_eccentricity`` (absolute, default 0.8) is called ``M``. Pass
        ``mitosis_min_eccentricity=None`` to instead use the ``mitosis_quantile``
        of the G2/M eccentricity distribution. Works for either H1.0 fluorophore
        (mNeonGreen or miRFP720), since both report chromatin morphology.

    Returns
    -------
    ``(df_with_phase, thresholds_used)``. The dataframe also carries boolean
    ``{marker}_pos`` columns for transparency.
    """
    truth_table = dict(truth_table or DEFAULT_TRUTH_TABLE)
    out = df.copy()

    roles = list(marker_cols.keys())
    for role in ("cdt1", "geminin", "slbp"):
        if role not in marker_cols:
            raise KeyError(f"marker_cols missing required role {role!r}")

    used: dict[str, float] = {}

    def gate_column(role: str, col: str) -> np.ndarray:
        if thresholds is not None and role in thresholds and np.isfinite(
            thresholds[role]
        ):
            thr = float(thresholds[role])
            used[role] = thr
            return out[col].to_numpy() > thr
        if per_frame:
            pos = np.zeros(len(out), dtype=bool)
            frame_thrs = []
            for _, fdf in out.groupby("frame"):
                thr = estimate_threshold(fdf[col].to_numpy(), method, log)
                frame_thrs.append(thr)
                pos[fdf.index] = fdf[col].to_numpy() > thr
            used[role] = float(np.nanmedian(frame_thrs))
            return pos
        thr = estimate_threshold(out[col].to_numpy(), method, log)
        used[role] = thr
        return out[col].to_numpy() > thr

    pos = {}
    for role in roles:
        col = marker_cols[role]
        if col not in out.columns:
            raise KeyError(f"Column {col!r} for marker {role!r} not in dataframe")
        pos[role] = gate_column(role, col)
        out[f"{role}_pos"] = pos[role]

    keys = list(zip(pos["cdt1"], pos["geminin"], pos["slbp"]))
    phases = np.array(
        [truth_table.get((bool(a), bool(b), bool(c)), "undetermined") for a, b, c in keys],
        dtype=object,
    )

    if detect_mitosis and eccentricity_col in out.columns:
        # Mitotic H1.0 chromatin condenses into an elongated/rectangular plate
        # -> HIGH eccentricity. Only consider the Geminin-high (G2/M) window.
        g2m = phases == "G2/M"
        if g2m.any():
            ecc_all = out[eccentricity_col].to_numpy(dtype=float)
            if mitosis_min_eccentricity is not None:
                ecc_thr = float(mitosis_min_eccentricity)
            else:
                ecc_thr = float(np.nanquantile(ecc_all[g2m], mitosis_quantile))
            mitotic = g2m & (ecc_all >= ecc_thr)
            phases[mitotic] = "M"
            used["mitosis_min_eccentricity"] = ecc_thr

    out[phase_col] = phases
    logger.info(
        "FUCCI-4 phase calls: %s",
        ", ".join(f"{k}={v}" for k, v in pd.Series(phases).value_counts().items()),
    )
    logger.info("Auto thresholds: %s", used)
    return out, used
