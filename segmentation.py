"""Single-channel nuclear segmentation via CellPose.

Segments the constitutive nuclear channel (e.g. miRFP720-H1.0 of the modified
FUCCI-4 line) frame by frame and returns an integer label movie. CellPose is
imported lazily so the rest of the pipeline (tracking, intensity, phase, VME)
can be imported and tested without it installed.

This deliberately segments a *single* channel rather than ConfluentFUCCI's
red+green merge: the constitutive H1.0 marker is present in every phase, so one
label set captures every nucleus every frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("fucci_vme.segmentation")


def segment_stack(
    stack: np.ndarray,
    diameter: Optional[float] = None,
    model_type: str = "nuclei",
    pretrained_model: Optional[str | Path] = None,
    gpu: bool = False,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
) -> np.ndarray:
    """Segment a ``(T, Y, X)`` single-channel stack into a label movie.

    Parameters
    ----------
    stack:
        2D ``(Y, X)`` or 3D ``(T, Y, X)`` intensity image of the nuclear channel.
    diameter:
        Approximate nuclear diameter in pixels (``None`` lets CellPose estimate).
    model_type:
        Built-in CellPose model name (e.g. ``'nuclei'``, ``'cyto3'``). Ignored if
        ``pretrained_model`` is given.
    pretrained_model:
        Path to a custom CellPose model (e.g. one fine-tuned on H1.0 nuclei).
    gpu:
        Use CUDA if available.

    Returns
    -------
    Integer label movie of shape ``(T, Y, X)`` (0 = background).
    """
    try:
        import cellpose
        from cellpose import models
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "cellpose is required for segmentation. Install with "
            "`pip install cellpose` (a CUDA-capable GPU is recommended)."
        ) from exc

    stack = np.asarray(stack)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(f"Expected (T, Y, X) or (Y, X); got {stack.shape}")

    # CellPose v4 ("Cellpose-SAM") removed the ``models.Cellpose`` class, the
    # ``model_type`` argument, and the per-call ``channels`` argument. Detect the
    # major version and build the model / eval kwargs accordingly so this works
    # on both v4 and the older v2/v3 API.
    try:
        from importlib.metadata import version as _pkg_version

        cp_major = int(_pkg_version("cellpose").split(".")[0])
    except Exception:  # pragma: no cover - fall back to attribute probing
        cp_major = 3 if hasattr(models, "Cellpose") else 4

    base_eval = dict(flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)

    if cp_major >= 4:
        # v4: single generalist model, grayscale input, no channels/model_type
        if pretrained_model is not None:
            model = models.CellposeModel(gpu=gpu, pretrained_model=str(pretrained_model))
        else:
            model = models.CellposeModel(gpu=gpu)
        eval_kwargs = dict(base_eval)
        if diameter is not None:
            eval_kwargs["diameter"] = diameter
    else:
        # v2/v3: explicit nuclei model + channels=[0, 0] for a single grayscale plane
        if pretrained_model is not None:
            model = models.CellposeModel(gpu=gpu, pretrained_model=str(pretrained_model))
        elif hasattr(models, "Cellpose"):
            model = models.Cellpose(gpu=gpu, model_type=model_type)
        else:  # pragma: no cover - unusual builds
            model = models.CellposeModel(gpu=gpu, model_type=model_type)
        eval_kwargs = dict(base_eval, channels=[0, 0], diameter=diameter)

    logger.info("Using cellpose v%s (major=%d)", getattr(cellpose, "version", "?"), cp_major)

    labels = np.zeros(stack.shape, dtype=np.int32)
    for t in range(stack.shape[0]):
        result = model.eval(stack[t].astype(np.float32), **eval_kwargs)
        masks = result[0]  # (masks, flows, styles[, diams])
        labels[t] = masks.astype(np.int32)
        logger.info("Segmented frame %d/%d: %d nuclei", t + 1, stack.shape[0], int(masks.max()))
    return labels
