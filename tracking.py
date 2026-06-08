"""Thin single-channel nuclear tracking for FUCCI-4 data.

This module tracks segmented nuclei across frames using a Linear Assignment
Problem (LAP) linker driven by mask overlap (IoU) with a centroid-distance
fallback. This is the same algorithm family as TrackMate's default LAP /
overlap trackers (Jaqaman et al. 2008), implemented with only ``scipy`` so it
needs no Java, Fiji, or Docker.

It is intended to run on a *single* constitutive nuclear channel (e.g. the
miRFP720-H1.0 marker of the modified FUCCI-4 line), which labels every nucleus
in every cell-cycle phase. That yields one complete, phase-independent label set
per frame -- exactly what reliable neighbor topology for VME analysis requires.

Region measurements (centroid, area, eccentricity) are computed with pure-numpy
``bincount`` moment accumulation, so ``scikit-image`` is not required.

Main entry point: :func:`track_masks`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("fucci_vme.tracking")

_FORBIDDEN = 1.0e6  # cost assigned to disallowed links


@dataclass
class TrackingConfig:
    """Parameters for the LAP overlap tracker and lineage assignment."""

    max_distance: float = 30.0  # max centroid displacement (pixels) for a link
    min_iou: float = 0.1  # minimum mask IoU to consider an overlap-based link
    max_gap: int = 1  # frames a track may disappear and still be reconnected
    min_track_length: int = 1  # drop tracks observed in fewer frames than this

    # --- division / lineage detection ---
    detect_divisions: bool = True
    # max distance (px) from a newborn track to its candidate mother at the
    # previous frame. None -> 2 * max_distance.
    division_max_distance: Optional[float] = None
    # a daughter's area must be within [min, max] * mother area at division
    min_daughter_area_ratio: float = 0.15
    max_daughter_area_ratio: float = 0.90
    # combined daughter area must be within (1 +/- tol) * mother area
    division_area_tol: float = 0.6


# --------------------------------------------------------------------------- #
# Region properties (pure numpy, label-wise moment accumulation)
# --------------------------------------------------------------------------- #
def regionprops_numpy(labels: np.ndarray) -> pd.DataFrame:
    """Compute per-label area, centroid and eccentricity for one label image.

    Parameters
    ----------
    labels:
        2D integer label image (0 = background).

    Returns
    -------
    DataFrame with columns ``[label, POSITION_X, POSITION_Y, AREA, eccentricity]``
    for every non-zero label present.
    """
    flat = labels.ravel()
    n = int(flat.max()) if flat.size else 0
    if n == 0:
        return pd.DataFrame(
            columns=["label", "POSITION_X", "POSITION_Y", "AREA", "eccentricity"]
        )

    ys, xs = np.indices(labels.shape)
    xs = xs.ravel().astype(np.float64)
    ys = ys.ravel().astype(np.float64)

    minlen = n + 1
    area = np.bincount(flat, minlength=minlen).astype(np.float64)
    sx = np.bincount(flat, weights=xs, minlength=minlen)
    sy = np.bincount(flat, weights=ys, minlength=minlen)
    sxx = np.bincount(flat, weights=xs * xs, minlength=minlen)
    syy = np.bincount(flat, weights=ys * ys, minlength=minlen)
    sxy = np.bincount(flat, weights=xs * ys, minlength=minlen)

    with np.errstate(invalid="ignore", divide="ignore"):
        cx = sx / area
        cy = sy / area
        mu_xx = sxx / area - cx * cx
        mu_yy = syy / area - cy * cy
        mu_xy = sxy / area - cx * cy

    # eccentricity from the eigenvalues of the second central moment matrix
    common = np.sqrt(np.clip((mu_xx - mu_yy) ** 2 + 4.0 * mu_xy**2, 0, None))
    l1 = 0.5 * (mu_xx + mu_yy + common)
    l2 = 0.5 * (mu_xx + mu_yy - common)
    with np.errstate(invalid="ignore", divide="ignore"):
        ecc = np.sqrt(np.clip(1.0 - (l2 / l1), 0, 1))
    ecc = np.where(l1 > 0, ecc, 0.0)

    labels_idx = np.arange(minlen)
    keep = (labels_idx > 0) & (area > 0)
    return pd.DataFrame(
        {
            "label": labels_idx[keep],
            "POSITION_X": cx[keep],
            "POSITION_Y": cy[keep],
            "AREA": area[keep],
            "eccentricity": ecc[keep],
        }
    )


# --------------------------------------------------------------------------- #
# Overlap between consecutive label frames
# --------------------------------------------------------------------------- #
def _overlap_counts(a: np.ndarray, b: np.ndarray) -> dict[tuple[int, int], int]:
    """Count shared pixels for every (label_a, label_b) pair where both > 0."""
    mask = (a > 0) & (b > 0)
    if not mask.any():
        return {}
    av = a[mask].astype(np.int64)
    bv = b[mask].astype(np.int64)
    stride = int(bv.max()) + 1
    keys = av * stride + bv
    uniq, counts = np.unique(keys, return_counts=True)
    return {(int(k // stride), int(k % stride)): int(c) for k, c in zip(uniq, counts)}


def _build_cost_matrix(
    prev_props: pd.DataFrame,
    cur_props: pd.DataFrame,
    prev_labels_img: np.ndarray,
    cur_labels_img: np.ndarray,
    cfg: TrackingConfig,
) -> np.ndarray:
    """Cost matrix (prev x cur). Lower cost = better link.

    Overlap-based links (IoU >= min_iou) get cost ``1 - IoU`` in [0, 1).
    Non-overlapping but nearby links (distance <= max_distance) get a higher
    cost in [1, 2). Everything else is forbidden.
    """
    p_labels = prev_props["label"].to_numpy()
    c_labels = cur_props["label"].to_numpy()
    p_area = dict(zip(p_labels, prev_props["AREA"].to_numpy()))
    c_area = dict(zip(c_labels, cur_props["AREA"].to_numpy()))
    p_idx = {lab: i for i, lab in enumerate(p_labels)}
    c_idx = {lab: j for j, lab in enumerate(c_labels)}

    cost = np.full((len(p_labels), len(c_labels)), _FORBIDDEN, dtype=np.float64)

    # distance-based fallback (vectorized)
    pxy = prev_props[["POSITION_X", "POSITION_Y"]].to_numpy()
    cxy = cur_props[["POSITION_X", "POSITION_Y"]].to_numpy()
    if len(pxy) and len(cxy):
        d = np.sqrt(
            ((pxy[:, None, :] - cxy[None, :, :]) ** 2).sum(axis=2)
        )
        near = d <= cfg.max_distance
        cost[near] = 1.0 + (d[near] / cfg.max_distance)

    # overlap-based links override the distance fallback when good enough
    overlaps = _overlap_counts(prev_labels_img, cur_labels_img)
    for (la, lb), inter in overlaps.items():
        if la not in p_idx or lb not in c_idx:
            continue
        union = p_area[la] + c_area[lb] - inter
        iou = inter / union if union > 0 else 0.0
        if iou >= cfg.min_iou:
            cost[p_idx[la], c_idx[lb]] = 1.0 - iou
    return cost


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
def track_masks(
    masks: np.ndarray,
    cfg: Optional[TrackingConfig] = None,
) -> pd.DataFrame:
    """Link segmented nuclei across frames into tracks.

    Parameters
    ----------
    masks:
        Integer label movie of shape ``(T, Y, X)`` (0 = background, per-frame
        labels need not be consistent across frames).
    cfg:
        :class:`TrackingConfig`. Defaults are reasonable for confluent monolayers.

    Returns
    -------
    DataFrame with one row per (frame, nucleus):
    ``[frame, label, track_id, cell_id, POSITION_X, POSITION_Y, AREA, eccentricity]``
    where ``track_id`` (and the string ``cell_id``) are stable across frames.
    """
    cfg = cfg or TrackingConfig()
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"masks must be (T, Y, X); got shape {masks.shape}")

    n_frames = masks.shape[0]
    props_per_frame = [regionprops_numpy(masks[t]) for t in range(n_frames)]

    records: list[pd.DataFrame] = []
    next_track_id = 0
    # mapping from a previous frame's label -> track id, kept for up to max_gap
    # frames so a momentarily-lost nucleus can be reconnected.
    history: list[dict[int, int]] = []  # most-recent-first list of {label: track_id}
    history_imgs: list[np.ndarray] = []
    history_props: list[pd.DataFrame] = []

    for t in range(n_frames):
        cur_props = props_per_frame[t].copy()
        if cur_props.empty:
            history.insert(0, {})
            history_imgs.insert(0, masks[t])
            history_props.insert(0, cur_props)
            continue

        cur_track_ids = {lab: None for lab in cur_props["label"].to_numpy()}
        linked_tracks: set = set()  # tracks already extended into this frame
        recent_tracks: set = set()  # tracks seen in a more recent history frame

        # Try to link against recent frames (gap closing), nearest in time first.
        for gap, (prev_map, prev_img, prev_props) in enumerate(
            zip(history, history_imgs, history_props)
        ):
            if gap > cfg.max_gap:
                break
            if prev_props.empty or "track_id" not in prev_props.columns:
                continue

            # A track that already has an observation more recently (or was
            # already extended into this frame) must not be reconnected from an
            # older frame -- otherwise a mother that continued into one daughter
            # would also "absorb" the second daughter across the gap.
            blocked = linked_tracks | recent_tracks
            eligible_prev = prev_props[~prev_props["track_id"].isin(blocked)]
            recent_tracks |= set(prev_props["track_id"])

            unlinked_cur = cur_props[
                cur_props["label"].map(lambda lab: cur_track_ids[lab] is None)
            ]
            if unlinked_cur.empty or eligible_prev.empty:
                continue

            cost = _build_cost_matrix(
                eligible_prev, unlinked_cur, prev_img, masks[t], cfg
            )
            row_ind, col_ind = linear_sum_assignment(cost)
            prev_labels = eligible_prev["label"].to_numpy()
            cur_labels = unlinked_cur["label"].to_numpy()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= _FORBIDDEN:
                    continue
                tid = prev_map.get(prev_labels[r])
                cur_label = cur_labels[c]
                if tid is not None and cur_track_ids[cur_label] is None:
                    cur_track_ids[cur_label] = tid
                    linked_tracks.add(tid)

        # start new tracks for anything still unlinked
        for lab in cur_props["label"].to_numpy():
            if cur_track_ids[lab] is None:
                cur_track_ids[lab] = next_track_id
                next_track_id += 1

        cur_props["track_id"] = cur_props["label"].map(cur_track_ids)
        cur_props["frame"] = t
        records.append(cur_props)

        history.insert(0, dict(zip(cur_props["label"], cur_props["track_id"])))
        history_imgs.insert(0, masks[t])
        history_props.insert(0, cur_props)
        # trim history beyond the gap window
        keep = cfg.max_gap + 1
        del history[keep:]
        del history_imgs[keep:]
        del history_props[keep:]

    out_cols = [
        "frame", "label", "track_id", "cell_id", "parent_track_id",
        "lineage_id", "generation", "POSITION_X", "POSITION_Y",
        "AREA", "eccentricity",
    ]
    if not records:
        return pd.DataFrame(columns=out_cols)

    df = pd.concat(records, ignore_index=True)

    if cfg.min_track_length > 1:
        lengths = df.groupby("track_id")["frame"].transform("size")
        df = df[lengths >= cfg.min_track_length].copy()

    n_raw_tracks = df["track_id"].nunique()

    if cfg.detect_divisions:
        divisions = detect_divisions(df, cfg)
        df = assign_lineage(df, divisions)
        logger.info(
            "Tracked %d nuclei; %d raw tracks -> %d lineage segments "
            "(%d divisions, %d founder lineages) across %d frames.",
            len(df), n_raw_tracks, df["track_id"].nunique(),
            len(divisions), df["lineage_id"].nunique(), n_frames,
        )
    else:
        df["parent_track_id"] = -1
        df["lineage_id"] = df["track_id"]
        df["generation"] = 0
        logger.info(
            "Tracked %d nuclei into %d tracks across %d frames (divisions off).",
            len(df), n_raw_tracks, n_frames,
        )

    df["cell_id"] = "trk_" + df["track_id"].astype(int).astype(str)
    df = df.sort_values(["lineage_id", "track_id", "frame"]).reset_index(drop=True)
    return df[out_cols]


# --------------------------------------------------------------------------- #
# Division detection + lineage assignment
# --------------------------------------------------------------------------- #
def detect_divisions(spots: pd.DataFrame, cfg: TrackingConfig) -> list[dict]:
    """Detect mitotic divisions among the 1<->1-linked track segments.

    A division is inferred when a track is *born* at frame ``t`` (> the movie
    start) next to a "mother" track that exists at both ``t-1`` and ``t`` (i.e.
    the 1<->1 linker followed the mother into one daughter and the other daughter
    started a fresh track), subject to distance and area-conservation checks.

    Returns a list of ``{"parent": raw_id, "daughter": raw_id, "frame": t}``.
    """
    if spots.empty:
        return []

    dmax = cfg.division_max_distance or (2.0 * cfg.max_distance)
    global_min = int(spots["frame"].min())
    rec = spots.set_index(["track_id", "frame"]).sort_index()
    by_frame = {int(f): sub for f, sub in spots.groupby("frame")}

    first_frame = spots.groupby("track_id")["frame"].min()
    born_segments = first_frame[first_frame > global_min].index.tolist()
    # sort newborns by birth frame for deterministic assignment
    born_segments.sort(key=lambda b: int(first_frame[b]))

    divisions: list[dict] = []
    used_parent_frame: set[tuple] = set()

    for daughter in born_segments:
        t = int(first_frame[daughter])
        prev = by_frame.get(t - 1)
        cur = by_frame.get(t)
        if prev is None or cur is None:
            continue

        bspot = rec.loc[(daughter, t)]
        bx, by_, ba = float(bspot.POSITION_X), float(bspot.POSITION_Y), float(bspot.AREA)

        # mother candidates: present at t-1 AND t (they "continue"), excluding self
        cont = set(prev["track_id"]).intersection(cur["track_id"])
        cont.discard(daughter)
        if not cont:
            continue
        prev_cont = prev[prev["track_id"].isin(cont)]
        dist = np.hypot(
            prev_cont["POSITION_X"].to_numpy() - bx,
            prev_cont["POSITION_Y"].to_numpy() - by_,
        )
        order = np.argsort(dist)
        for oi in order:
            d = float(dist[oi])
            if d > dmax:
                break
            row = prev_cont.iloc[int(oi)]
            parent = row["track_id"]
            if (parent, t) in used_parent_frame:
                continue
            parent_area_prev = float(row["AREA"])
            if parent_area_prev <= 0:
                continue
            if not (
                cfg.min_daughter_area_ratio
                <= ba / parent_area_prev
                <= cfg.max_daughter_area_ratio
            ):
                continue
            try:
                cont_area_t = float(rec.loc[(parent, t)].AREA)
            except KeyError:
                continue
            combined = ba + cont_area_t
            lo = parent_area_prev * (1.0 - cfg.division_area_tol)
            hi = parent_area_prev * (1.0 + cfg.division_area_tol)
            if not (lo <= combined <= hi):
                continue
            divisions.append({"parent": parent, "daughter": daughter, "frame": t})
            used_parent_frame.add((parent, t))
            break

    return divisions


def assign_lineage(spots: pd.DataFrame, divisions: list[dict]) -> pd.DataFrame:
    """Split mother tracks at divisions and assign lineage metadata.

    Adds/overwrites ``track_id`` (segments that do not span a division),
    ``parent_track_id`` (-1 for founders), ``lineage_id`` (root founder id), and
    ``generation`` (0 for founders, +1 per division).
    """
    spots = spots.copy()
    if not divisions:
        spots["parent_track_id"] = -1
        spots["lineage_id"] = spots["track_id"]
        spots["generation"] = 0
        return spots

    cut_frames: dict[int, set] = defaultdict(set)
    born_parent: dict[int, tuple] = {}
    for dv in divisions:
        cut_frames[dv["parent"]].add(int(dv["frame"]))
        born_parent[dv["daughter"]] = (dv["parent"], int(dv["frame"]))

    # Build contiguous intervals per raw segment, cutting at division frames.
    next_id = 0
    interval_index: dict[int, list[tuple]] = {}  # raw -> [(start, end_excl, new_id)]
    parent_of: dict[int, Optional[int]] = {}

    for raw, sub in spots.groupby("track_id"):
        frames = sorted(int(f) for f in sub["frame"].unique())
        cuts = sorted(f for f in cut_frames.get(raw, ()) if frames[0] < f <= frames[-1])
        bounds = [frames[0]] + cuts + [frames[-1] + 1]
        ivs = []
        prev_new = None
        for k in range(len(bounds) - 1):
            nid = next_id
            next_id += 1
            parent_of[nid] = prev_new  # chained continuation
            ivs.append((bounds[k], bounds[k + 1], nid))
            prev_new = nid
        interval_index[int(raw)] = ivs

    def chunk_of(raw: int, frame: int) -> Optional[int]:
        for s, e, nid in interval_index.get(int(raw), ()):
            if s <= frame < e:
                return nid
        return None

    # Born daughters: their first segment's parent is the mother's segment at t-1.
    for daughter_raw, (parent_raw, t) in born_parent.items():
        ivs = interval_index.get(int(daughter_raw))
        if not ivs:
            continue
        parent_of[ivs[0][2]] = chunk_of(parent_raw, t - 1)

    # Resolve roots and generations (iterative walk up the parent chain).
    root_cache: dict[int, int] = {}
    gen_cache: dict[int, int] = {}

    def resolve(nid: int) -> tuple[int, int]:
        chain = []
        cur = nid
        while cur is not None and cur not in root_cache:
            chain.append(cur)
            cur = parent_of.get(cur)
        if cur is None:  # reached a founder
            base_root = chain[-1]
            base_gen = 0
            root_cache[base_root] = base_root
            gen_cache[base_root] = 0
            chain = chain[:-1]
            ref_gen = 0
            ref_root = base_root
        else:
            ref_root = root_cache[cur]
            ref_gen = gen_cache[cur]
        for node in reversed(chain):
            ref_gen += 1
            root_cache[node] = ref_root
            gen_cache[node] = ref_gen
        return root_cache[nid], gen_cache[nid]

    raws = spots["track_id"].to_numpy()
    frs = spots["frame"].to_numpy()
    new_track = np.empty(len(spots), dtype=np.int64)
    parent_track = np.empty(len(spots), dtype=np.int64)
    lineage = np.empty(len(spots), dtype=np.int64)
    generation = np.empty(len(spots), dtype=np.int64)

    for i in range(len(spots)):
        nid = chunk_of(int(raws[i]), int(frs[i]))
        new_track[i] = nid
        pid = parent_of.get(nid)
        parent_track[i] = -1 if pid is None else pid
        root, gen = resolve(nid)
        lineage[i] = root
        generation[i] = gen

    spots["track_id"] = new_track
    spots["parent_track_id"] = parent_track
    spots["lineage_id"] = lineage
    spots["generation"] = generation
    return spots
