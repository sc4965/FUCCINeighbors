# FUCCI Neighbors — Virus Microenvironment (VME) analysis

Tools for analyzing the **Virus Microenvironment (VME)** in live-imaging
experiments of a modified **FUCCI-4** cell line co-imaged with **GFP-HCMV**.

A **VME** is the set of *uninfected* cells immediately adjacent to an infected
cell at a given moment in time. Because infection can involve several touching
cells, contiguous infected cells are grouped into a single VME via connected
components of the per-frame Delaunay graph. The goal is to define VMEs
dynamically and track the cell-cycle fate of neighbors (e.g. G2/M arrest)
versus the distal control population.

There are **two ways to use this repository**:

1. **Self-contained pipeline** (`pipeline.py`) — channel-split TIFFs all the way
   to VME plots, with **no dependency on ConfluentFUCCI**. Uses CellPose for
   single-channel nuclear segmentation and a lightweight scipy LAP tracker.
2. **Post-processing only** (`fucci_vme.py`) — if you already have a tracked
   dataframe (e.g. ConfluentFUCCI's `confluent_fucci_data.csv`), apply just the
   VME spatial logic + plots.

---

## 1. Self-contained pipeline (recommended)

```
channel-split TIFFs
  ─(1)─►  segment constitutive nuclear channel (CellPose)      segmentation.py
  ─(2)─►  track nuclei, single-channel LAP overlap (scipy)     tracking.py
  ─(3)─►  per-nucleus marker + GFP intensities (numpy)         intensity.py
  ─(4)─►  auto-gated FUCCI-4 phases (G0/G1/G1S/S/G2M[/M])       phase_fucci4.py
  ─(5)─►  tag VMEs + plot phase trajectories                   fucci_vme.py
```

### Two infection-detection modes (two FUCCI-4 variants)

This experiment uses one of two FUCCI-4 lines, which differ only in where the
constitutive H1.0 nuclear marker lives and therefore how infection is detected.
Select with `--infection-mode`:

| `--infection-mode` | FUCCI-4 line | Nuclear marker | How infection is found |
| --- | --- | --- | --- |
| `intensity` | GFP-channel-free | **miRFP720–H1.0** | a **separate GFP-HCMV channel** thresholded by intensity |
| `morphology` | mNeonGreen-H1.0 | **mNeonGreen–H1.0** (green) | **shape**: cell-surface GFP shares the green channel, so infected cells show a **large, non-round** green object vs the small round nuclear dot |

In both cases the three FUCCI phase markers are unchanged: mScarlet3–Cdt1 (G1/G0),
emiRFP670–Geminin (S/G2/M), mTagBFP2–SLBP (S).

### Why single-channel segmentation?

ConfluentFUCCI segments the **red and green channels separately and merges**
them, because classic 2-color FUCCI has no constitutive nuclear marker. Both
FUCCI-4 lines here include a constitutive H1.0 histone marker present in *every*
phase, so we segment that **one** nuclear channel, yielding a complete,
phase-independent label set every frame — exactly what reliable VME neighbor
topology needs.

### Morphology-based infection (mNeonGreen-H1.0 line)

When GFP-HCMV is expressed as a **cell-surface** protein in the same green
channel as mNeonGreen-H1.0, infection cannot be called by intensity (the nuclei
are green too). Instead `morphology.py` segments the green channel into objects
and classifies each by **shape**: an object that is **large** (area greater than
`--infection-area-factor` × the median object area) **and non-round**
(`circularity < --infection-min-circularity`, or `solidity <
--infection-min-solidity`) is infected cell-surface GFP. A tracked nucleus is
then flagged infected when ≥ `--infection-min-overlap` of its mask is covered by
infected green pixels (`--infection-assign overlap`, robust against clipping
neighbors) or, alternatively, when its centroid sits inside an infected object
(`--infection-assign centroid`). All thresholds are auto-seeded and tunable.

```bash
python pipeline.py --infection-mode morphology \
    --nuclear-tif mneongreen_h1.tif \
    --cdt1-tif mscarlet3.tif --geminin-tif emirfp670.tif --slbp-tif mtagbfp2.tif \
    --diameter 18 --gpu \
    --out-csv vme_tagged.csv --out-fig-prefix vme
# (in morphology mode the green/nuclear channel doubles as the infection channel,
#  so no --gfp-tif is needed; override with --green-tif if it lives in a separate file)

# no-dependency synthetic demo:
python pipeline.py --demo --infection-mode morphology --out-fig-prefix demo_morph
```

### FUCCI-4 marker → phase mapping

| Channel (marker)            | Reports          | High in   |
| --------------------------- | ---------------- | --------- |
| mScarlet3 – Cdt1(30–120)    | APC/C activity   | **G1/G0** |
| emiRFP670 – Geminin(1–110)  | SCF activity     | **S/G2/M**|
| mTagBFP2 – SLBP(18–126)     | replication      | **S**     |
| miRFP720 – H1.0             | constitutive     | all (segmentation / mitosis) |

Phases are assigned from the (Cdt1⁺, Geminin⁺, SLBP⁺) boolean combination via an
editable truth table in `phase_fucci4.DEFAULT_TRUTH_TABLE`.

### Install

```bash
pip install -r requirements.txt
```

`cellpose` and `tifffile` are only needed for stages 1–2 from real TIFFs; they
are imported lazily, so tracking/intensity/phase/VME work without them.

### Run

```bash
python pipeline.py \
    --nuclear-tif  h1_mirfp720.tif \
    --cdt1-tif     mscarlet3.tif \
    --geminin-tif  emirfp670.tif \
    --slbp-tif     mtagbfp2.tif \
    --gfp-tif      gfp_hcmv.tif \
    --gfp-threshold auto \
    --diameter 18 --gpu \
    --out-csv vme_tagged.csv --out-fig-prefix vme
```

Try it with **no imaging dependencies** on synthetic data:

```bash
python pipeline.py --demo --out-fig-prefix demo
```

Useful flags:

- `--masks-tif masks.tif` — supply a precomputed label movie and skip CellPose.
- `--pretrained-model PATH` — use a CellPose model fine-tuned on your H1.0 nuclei.
- `--gate-method {otsu,gmm,quantile}` — marker auto-gating method (default `otsu`).
- `--per-frame-gating` — re-estimate thresholds per frame (handles bleaching).
- `--gfp-threshold auto|<number>` — `auto` Otsu-gates GFP positivity.
- `--detect-mitosis` — within the Geminin-high (`G2/M`) window, reclassify cells
  with **condensed chromatin** to `M`. The H1.0 marker (mNeonGreen or miRFP720)
  forms an elongated/rectangular metaphase plate during mitosis, so high nuclear
  eccentricity flags it. Tune with `--mitosis-min-eccentricity` (default `0.8`,
  absolute); pass a negative value to instead gate on the `--mitosis-quantile`
  of the `G2/M` eccentricity distribution.
- `--max-distance` — tracking link radius in pixels.
- `--no-divisions` — disable lineage/division detection (plain 1↔1 tracks).
- `--division-max-distance` — max mother→daughter distance at a division (px;
  default `2 × max-distance`).

### Lineage / division tracking

By default the tracker detects mitotic divisions and builds a lineage tree. When
a cell divides, the **mother segment ends** and **both daughters receive fresh
`track_id`s** that point back to the mother via `parent_track_id` (the standard
mother→daughters convention). Each cell therefore has a trajectory that does not
span a division — which is biologically correct, since a daughter is a new cell
with its own fate. A division is inferred when a track is *born* next to a
mother track that continues through the same frame, subject to distance and
area-conservation checks (`min/max_daughter_area_ratio`, `division_area_tol` in
`TrackingConfig`). VME analysis benefits automatically: a daughter that ends up
adjacent to an infected cell is tracked as its own VME member.

### Auto-gating, then tuning

Thresholds are estimated automatically (Otsu in log space by default) and
printed at the end of a run. Once you have rough per-marker thresholds, override
them from Python:

```python
from pipeline import run_from_masks
tagged, info = run_from_masks(
    masks, marker_channels, gfp_channel=gfp,
    phase_thresholds={"cdt1": 120, "geminin": 90, "slbp": 110},  # tuned values
    gfp_threshold=300,
)
print(info["phase_thresholds"])  # what was actually used
```

### From Python (in-memory arrays)

```python
import numpy as np
from pipeline import run_from_masks

masks = ...            # (T, Y, X) int label movie
marker_channels = {"cdt1": ..., "geminin": ..., "slbp": ...}   # each (T, Y, X)
gfp = ...              # (T, Y, X)

tagged, info = run_from_masks(masks, marker_channels, gfp_channel=gfp,
                              gfp_threshold="auto")
```

---

## 2. Post-processing an existing tracked dataframe

If you already have a tracked dataframe, `fucci_vme.py` applies just the VME
logic. It auto-detects common column names and is configurable via `VMEConfig`
(`phase_col`, `phase_order`, `gfp_col`, `id_col`, ...), so it supports both the
2-color ConfluentFUCCI `color` scheme and an N-state FUCCI-4 `phase` column.

```python
import pandas as pd
from fucci_vme import VMEConfig, run_vme_analysis, plot_phase_trajectories

df = pd.read_csv("confluent_fucci_data.csv")
cfg = VMEConfig(gfp_col="MEAN_INTENSITY_CH3", gfp_threshold=500.0)
tagged, index_cells = run_vme_analysis(df, cfg)
plot_phase_trajectories(tagged, cfg, out_prefix="vme")
```

```bash
python fucci_vme.py --demo --out-fig-prefix demo   # 2-color synthetic example
```

---

## Output columns

The tagged dataframe adds:

| column                  | meaning                                                     |
| ----------------------- | ----------------------------------------------------------- |
| `track_id`              | lineage segment id (does not span a division)               |
| `cell_id`               | `trk_{track_id}` string identity used by the VME logic       |
| `parent_track_id`       | mother segment id (`-1` for founder cells)                  |
| `lineage_id`            | root founder id shared by a whole lineage tree              |
| `generation`            | 0 for founders, +1 per division                             |
| `phase`                 | FUCCI-4 phase call (G0/G1/G1S/S/G2M[/M]) — pipeline only     |
| `{marker}_mean`         | per-nucleus mean intensity (cdt1/geminin/slbp/gfp)          |
| `{marker}_pos`          | boolean auto-gate calls used for classification            |
| `is_infected`           | infected at that frame (GFP threshold or green-shape, by mode)|
| `is_index`              | first-crossing (index) cell, at/after its infection frame   |
| `is_vme`                | uninfected neighbor of an infected (index-containing) seed  |
| `vme_id`                | per-frame id of the contiguous VME the cell borders         |
| `vme_index_id`          | stable VME id = the sorted index cells in the seed          |
| `frame_since_infection` | frame minus the earliest global infection frame            |

The figure (`*_phase_trajectories.png`) has three panels: mean cell-cycle phase
over time (VME vs control, ± SEM) and stacked phase-fraction composition for the
VME and control populations.

---

## Module map

| file              | role                                                          |
| ----------------- | ------------------------------------------------------------- |
| `pipeline.py`     | end-to-end orchestration + CLI + synthetic `--demo`           |
| `segmentation.py` | lazy CellPose single-channel nuclear segmentation             |
| `tracking.py`     | scipy LAP overlap tracker, lineage/division detection, numpy region props |
| `intensity.py`    | per-nucleus mean/total intensity for any channel (numpy)      |
| `phase_fucci4.py` | auto-gating (Otsu/GMM/quantile) FUCCI-4 phase classifier       |
| `morphology.py`   | shape descriptors + morphology-based (cell-surface GFP) infection detection |
| `fucci_vme.py`    | Delaunay VME tagging, contiguity, and trajectory plots         |

## Caveats & assumptions

- **Tracking quality drives VME quality.** VME membership and trajectories are
  only as good as the segmentation/tracking; ID swaps corrupt trajectories. Tune
  `--max-distance` and `TrackingConfig.min_iou`/`max_gap` for your frame rate and
  motility. Division detection is heuristic (distance + area conservation);
  inspect `parent_track_id`/`generation` and tune the `*_daughter_area_ratio` /
  `division_area_tol` thresholds, or disable with `--no-divisions`.
- **G0 vs G1** cannot be separated from a single snapshot of markers; `G0` is
  emitted only when all markers are low. Dwell-time analysis can refine this.
- **Mitosis (M)** is inferred from H1.0 chromatin morphology: condensed mitotic
  chromatin is elongated/rectangular (high eccentricity) rather than a round
  interphase nucleus, and M occurs inside the Geminin-high window so detection is
  restricted to `G2/M`. This is a single-feature heuristic — validate
  `--mitosis-min-eccentricity` on your data (consider also extent/solidity for
  rectangularity), or leave it folded into `G2/M` by omitting `--detect-mitosis`.
  Note this is distinct from infection: condensed chromatin is nucleus-sized, so
  the morphology infection detector's large-area gate keeps mitotic cells from
  being mislabeled as infected.
- Channels must be **spatially registered** to the nuclear channel (same Y,X).
