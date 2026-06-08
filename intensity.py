"""Per-nucleus intensity readout for FUCCI-4 markers and the GFP-HCMV channel.

Given the label movie used for tracking and one or more co-registered intensity
channels, this computes the mean (and integrated) intensity inside every
nucleus, per frame, and merges those values onto the tracked spots dataframe.

This is the single operation that supplies *both* the FUCCI-4 phase markers
(mScarlet3-Cdt1, emiRFP670-Geminin, mTagBFP2-SLBP) and the separate GFP-HCMV
infection channel -- they are just different intensity stacks measured against
the same nuclear masks.

Implemented with pure-numpy ``bincount`` accumulation (no scikit-image).
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger("fucci_vme.intensity")


def _label_means(labels: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-label mean, per-label sum) indexed by label id (0..max)."""
    flat = labels.ravel()
    n = int(flat.max()) if flat.size else 0
    minlen = n + 1
    area = np.bincount(flat, minlength=minlen).astype(np.float64)
    total = np.bincount(flat, weights=image.ravel().astype(np.float64), minlength=minlen)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(area > 0, total / area, np.nan)
    return mean, total


def measure_channel_intensities(
    masks: np.ndarray,
    channels: Mapping[str, np.ndarray],
    spots_df: pd.DataFrame,
    include_total: bool = True,
) -> pd.DataFrame:
    """Measure mean intensity of each channel inside each tracked nucleus.

    Parameters
    ----------
    masks:
        Integer label movie ``(T, Y, X)`` used during tracking.
    channels:
        Mapping ``name -> stack`` where each stack is ``(T, Y, X)`` and spatially
        registered to ``masks``. ``name`` becomes the column prefix, e.g. a key
        ``"cdt1"`` yields a ``"cdt1_mean"`` column.
    spots_df:
        Tracked spots dataframe from :func:`tracking.track_masks`, requiring at
        least ``frame`` and ``label`` columns.
    include_total:
        Also emit ``{name}_total`` (integrated intensity) columns.

    Returns
    -------
    Copy of ``spots_df`` with ``{name}_mean`` (and optionally ``{name}_total``)
    columns added.
    """
    masks = np.asarray(masks)
    out = spots_df.copy()
    for name, stack in channels.items():
        stack = np.asarray(stack)
        if stack.shape != masks.shape:
            raise ValueError(
                f"Channel {name!r} shape {stack.shape} != masks shape {masks.shape}"
            )
        mean_col = f"{name}_mean"
        total_col = f"{name}_total"
        out[mean_col] = np.nan
        if include_total:
            out[total_col] = np.nan

        for t, frame_df in out.groupby("frame"):
            mean, total = _label_means(masks[int(t)], stack[int(t)])
            labels = frame_df["label"].to_numpy().astype(int)
            valid = labels < len(mean)
            idx = frame_df.index.to_numpy()
            vals = np.full(len(labels), np.nan)
            vals[valid] = mean[labels[valid]]
            out.loc[idx, mean_col] = vals
            if include_total:
                tvals = np.full(len(labels), np.nan)
                tvals[valid] = total[labels[valid]]
                out.loc[idx, total_col] = tvals

        logger.info("Measured channel %r -> %s", name, mean_col)
    return out
