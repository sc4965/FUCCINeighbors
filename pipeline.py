"""End-to-end FUCCI-4 + GFP-HCMV VME pipeline (no ConfluentFUCCI required).

Stages
------
    channel-split TIFFs
        -> (1) segment the constitutive nuclear channel (CellPose)   [segmentation.py]
        -> (2) track nuclei single-channel (LAP overlap)             [tracking.py]
        -> (3) measure per-nucleus marker + GFP intensities          [intensity.py]
        -> (4) auto-gate FUCCI-4 phases (G0/G1/G1S/S/G2M[/M])         [phase_fucci4.py]
        -> (5) tag Virus Microenvironments + plot trajectories       [fucci_vme.py]

Run on real data::

    python pipeline.py \
        --nuclear-tif h1.tif \
        --cdt1-tif mscarlet3.tif --geminin-tif emirfp670.tif --slbp-tif mtagbfp2.tif \
        --gfp-tif gfp.tif \
        --gfp-threshold auto \
        --diameter 18 --gpu \
        --out-csv vme_tagged.csv --out-fig-prefix vme

Try it without any imaging deps::

    python pipeline.py --demo --out-fig-prefix demo
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

import phase_fucci4
from fucci_vme import VMEConfig, plot_phase_trajectories, tag_vme
from intensity import measure_channel_intensities
from phase_fucci4 import classify_fucci4
from tracking import TrackingConfig, track_masks

logger = logging.getLogger("fucci_vme.pipeline")

DEFAULT_MARKER_MAP = {"cdt1": "cdt1", "geminin": "geminin", "slbp": "slbp"}


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_tiff(path: str | Path) -> np.ndarray:
    """Load a (T, Y, X) stack from a TIFF (lazy tifffile import)."""
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover
        raise ImportError("tifffile is required to read TIFFs (`pip install tifffile`).") from exc
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


# --------------------------------------------------------------------------- #
# Core pipeline (operates on in-memory arrays -> testable without CellPose)
# --------------------------------------------------------------------------- #
def run_from_masks(
    masks: np.ndarray,
    marker_channels: Mapping[str, np.ndarray],
    gfp_channel: Optional[np.ndarray] = None,
    marker_map: Mapping[str, str] = DEFAULT_MARKER_MAP,
    gfp_threshold: float | str = "auto",
    tracking_cfg: Optional[TrackingConfig] = None,
    gate_method: str = "otsu",
    per_frame_gating: bool = False,
    detect_mitosis: bool = False,
    phase_thresholds: Optional[Mapping[str, float]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Track -> measure intensities -> classify phase -> tag VME.

    ``marker_channels`` maps a *channel name* to its ``(T, Y, X)`` stack; the keys
    must include the channel names referenced by ``marker_map`` values, plus any
    others you want measured. ``gfp_channel`` is the separate GFP-HCMV stack.

    Returns ``(tagged_df, info)`` where ``info`` carries the index cells and the
    gating thresholds actually used.
    """
    spots = track_masks(masks, tracking_cfg)
    if spots.empty:
        raise RuntimeError("Tracking produced no spots; check segmentation masks.")

    channels = dict(marker_channels)
    if gfp_channel is not None:
        channels["gfp"] = gfp_channel
    spots = measure_channel_intensities(masks, channels, spots)

    marker_cols = {role: f"{ch}_mean" for role, ch in marker_map.items()}
    spots, thresholds = classify_fucci4(
        spots,
        marker_cols=marker_cols,
        method=gate_method,
        per_frame=per_frame_gating,
        detect_mitosis=detect_mitosis,
        thresholds=phase_thresholds,
    )

    # Resolve the GFP infection threshold (numeric or auto-gated).
    if gfp_channel is not None:
        if isinstance(gfp_threshold, str) and gfp_threshold == "auto":
            gfp_thr = phase_fucci4.estimate_threshold(
                spots["gfp_mean"].to_numpy(), method=gate_method, log=True
            )
            logger.info("Auto GFP infection threshold: %.4g", gfp_thr)
        else:
            gfp_thr = float(gfp_threshold)
    else:
        gfp_thr = float("inf")

    cfg = VMEConfig(
        phase_col="phase",
        gfp_col="gfp_mean" if gfp_channel is not None else None,
        id_col="cell_id",
        gfp_threshold=gfp_thr,
        phase_order=phase_fucci4.DEFAULT_PHASE_ORDER,
        phase_labels=dict(phase_fucci4.DEFAULT_PHASE_LABELS),
    )
    tagged, index_cells = tag_vme(spots, cfg, id_col="cell_id")

    info = {
        "index_cells": index_cells,
        "phase_thresholds": thresholds,
        "gfp_threshold": gfp_thr,
        "config": cfg,
    }
    return tagged, info


def run_from_tiffs(
    nuclear_tif: str | Path,
    marker_tifs: Mapping[str, str | Path],
    gfp_tif: Optional[str | Path] = None,
    masks_tif: Optional[str | Path] = None,
    diameter: Optional[float] = None,
    pretrained_model: Optional[str | Path] = None,
    gpu: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Disk-based entry point: load TIFFs, segment, then :func:`run_from_masks`."""
    if masks_tif is not None:
        masks = load_tiff(masks_tif).astype(np.int32)
        logger.info("Loaded precomputed masks from %s", masks_tif)
    else:
        from segmentation import segment_stack

        nuclear = load_tiff(nuclear_tif)
        masks = segment_stack(
            nuclear, diameter=diameter, pretrained_model=pretrained_model, gpu=gpu
        )

    marker_channels = {name: load_tiff(path) for name, path in marker_tifs.items()}
    gfp_channel = load_tiff(gfp_tif) if gfp_tif is not None else None
    return run_from_masks(masks, marker_channels, gfp_channel, **kwargs)


# --------------------------------------------------------------------------- #
# Synthetic demo (no CellPose / tifffile needed)
# --------------------------------------------------------------------------- #
def generate_synthetic_movie(
    n_frames: int = 24,
    grid: int = 7,
    spacing: int = 28,
    radius: int = 9,
    infection_frame: int = 5,
    seed: int = 0,
):
    """Build a synthetic label movie + marker/GFP channels for testing.

    Returns ``(masks, marker_channels, gfp_channel)``. The central nucleus turns
    GFP-positive at ``infection_frame``; its ring neighbors are biased toward a
    G2/M arrest afterward to mimic VME dysregulation.
    """
    rng = np.random.default_rng(seed)
    size = (grid + 1) * spacing
    masks = np.zeros((n_frames, size, size), dtype=np.int32)
    cdt1 = np.zeros_like(masks, dtype=np.float32)
    gem = np.zeros_like(masks, dtype=np.float32)
    slbp = np.zeros_like(masks, dtype=np.float32)
    gfp = np.zeros_like(masks, dtype=np.float32)

    yy, xx = np.indices((size, size))
    center = (grid // 2, grid // 2)
    phase_cycle = ["G1", "G1/S", "S", "G2/M"]

    def marker_levels(phase: str):
        base = {"cdt1": 30.0, "geminin": 30.0, "slbp": 30.0}
        if phase == "G1":
            base["cdt1"] = 400.0
        elif phase == "G1/S":
            base["cdt1"] = 350.0; base["slbp"] = 300.0
        elif phase == "S":
            base["slbp"] = 450.0; base["geminin"] = 250.0
        elif phase == "G2/M":
            base["geminin"] = 450.0
        return base

    cell_idx = 0
    for i in range(grid):
        for j in range(grid):
            cell_idx += 1
            cx = (i + 1) * spacing
            cy = (j + 1) * spacing
            is_center = (i, j) == center
            is_ring = max(abs(i - center[0]), abs(j - center[1])) == 1
            cursor = int(rng.integers(0, len(phase_cycle)))
            for f in range(n_frames):
                if rng.random() < 0.3:
                    cursor = (cursor + 1) % len(phase_cycle)
                phase = phase_cycle[cursor]
                if is_ring and f >= infection_frame and rng.random() < 0.7:
                    phase = "G2/M"  # mimic arrest in the VME
                jx = int(rng.normal(0, 1))
                jy = int(rng.normal(0, 1))
                disk = (xx - (cx + jx)) ** 2 + (yy - (cy + jy)) ** 2 <= radius**2
                masks[f][disk] = cell_idx
                lv = marker_levels(phase)
                cdt1[f][disk] = lv["cdt1"] + rng.normal(0, 10)
                gem[f][disk] = lv["geminin"] + rng.normal(0, 10)
                slbp[f][disk] = lv["slbp"] + rng.normal(0, 10)
                if is_center and f >= infection_frame:
                    gfp[f][disk] = 600.0 + 40.0 * (f - infection_frame) + rng.normal(0, 20)
                else:
                    gfp[f][disk] = rng.normal(40, 10)

    marker_channels = {"cdt1": cdt1, "geminin": gem, "slbp": slbp}
    return masks, marker_channels, gfp


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FUCCI-4 + GFP-HCMV Virus Microenvironment pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--demo", action="store_true", help="Run on synthetic data.")
    p.add_argument("--nuclear-tif", type=str, help="Constitutive nuclear (H1.0/miRFP720) stack.")
    p.add_argument("--cdt1-tif", type=str, help="mScarlet3-Cdt1 stack.")
    p.add_argument("--geminin-tif", type=str, help="emiRFP670-Geminin stack.")
    p.add_argument("--slbp-tif", type=str, help="mTagBFP2-SLBP stack.")
    p.add_argument("--gfp-tif", type=str, help="GFP-HCMV stack.")
    p.add_argument("--masks-tif", type=str, help="Precomputed label movie (skips CellPose).")
    p.add_argument("--diameter", type=float, default=None, help="Nuclear diameter (px).")
    p.add_argument("--pretrained-model", type=str, default=None, help="Custom CellPose model.")
    p.add_argument("--gpu", action="store_true", help="Use GPU for CellPose.")
    p.add_argument("--gfp-threshold", type=str, default="auto", help="Number or 'auto'.")
    p.add_argument("--gate-method", type=str, default="otsu", choices=["otsu", "gmm", "quantile"])
    p.add_argument("--per-frame-gating", action="store_true")
    p.add_argument("--detect-mitosis", action="store_true")
    p.add_argument("--max-distance", type=float, default=30.0, help="Tracking link radius (px).")
    p.add_argument("--no-divisions", action="store_true", help="Disable lineage/division detection.")
    p.add_argument("--division-max-distance", type=float, default=None,
                   help="Max mother->daughter distance at division (px); default 2*max-distance.")
    p.add_argument("--out-csv", type=str, default=None)
    p.add_argument("--out-fig-prefix", type=str, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    tracking_cfg = TrackingConfig(
        max_distance=args.max_distance,
        detect_divisions=not args.no_divisions,
        division_max_distance=args.division_max_distance,
    )

    if args.demo or not args.nuclear_tif:
        if not args.demo:
            logger.warning("No --nuclear-tif given; running --demo on synthetic data.")
        masks, marker_channels, gfp = generate_synthetic_movie()
        tagged, info = run_from_masks(
            masks,
            marker_channels,
            gfp_channel=gfp,
            gfp_threshold=(args.gfp_threshold if args.gfp_threshold != "auto" else "auto"),
            tracking_cfg=tracking_cfg,
            gate_method=args.gate_method,
            per_frame_gating=args.per_frame_gating,
            detect_mitosis=args.detect_mitosis,
        )
    else:
        marker_tifs = {}
        if args.cdt1_tif:
            marker_tifs["cdt1"] = args.cdt1_tif
        if args.geminin_tif:
            marker_tifs["geminin"] = args.geminin_tif
        if args.slbp_tif:
            marker_tifs["slbp"] = args.slbp_tif
        missing = {"cdt1", "geminin", "slbp"} - set(marker_tifs)
        if missing:
            raise SystemExit(f"Missing required marker TIFFs: {sorted(missing)}")
        gfp_threshold = args.gfp_threshold
        try:
            gfp_threshold = float(gfp_threshold)
        except ValueError:
            pass  # keep 'auto'
        tagged, info = run_from_tiffs(
            nuclear_tif=args.nuclear_tif,
            marker_tifs=marker_tifs,
            gfp_tif=args.gfp_tif,
            masks_tif=args.masks_tif,
            diameter=args.diameter,
            pretrained_model=args.pretrained_model,
            gpu=args.gpu,
            gfp_threshold=gfp_threshold,
            tracking_cfg=tracking_cfg,
            gate_method=args.gate_method,
            per_frame_gating=args.per_frame_gating,
            detect_mitosis=args.detect_mitosis,
        )

    index_cells = info["index_cells"]
    print("\n=== Auto-gated marker thresholds ===")
    for k, v in info["phase_thresholds"].items():
        print(f"  {k:24s} {v:.4g}")
    print(f"  {'gfp (infection)':24s} {info['gfp_threshold']:.4g}")

    print("\n=== Index (infected) cells ===")
    print("  none detected" if index_cells.empty else index_cells.to_string(index=False))

    if "parent_track_id" in tagged.columns:
        n_div = int((tagged.groupby("track_id")["parent_track_id"].first() >= 0).sum())
        n_lineages = tagged["lineage_id"].nunique()
        n_segments = tagged["track_id"].nunique()
        print(
            f"\nLineage: {n_segments} track segments, {n_lineages} founder lineages, "
            f"{n_div} division(s) detected."
        )

    vme_rows = tagged[tagged["is_vme"]]
    print(f"\nVME observations: {len(vme_rows)}  |  unique VME cells: {vme_rows['cell_id'].nunique()}")
    print("\nPhase distribution (VME vs control):")
    print(
        tagged.assign(group=np.where(tagged["is_vme"], "VME", "control"))
        .groupby(["group", "phase"]).size().unstack(fill_value=0)
    )

    if args.out_csv:
        tagged.to_csv(args.out_csv, index=False)
        logger.info("Wrote %s", args.out_csv)

    if args.out_fig_prefix or args.show:
        plot_phase_trajectories(tagged, info["config"], out_prefix=args.out_fig_prefix, show=args.show)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
