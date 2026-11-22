# Post_masks: snow and desert/beach revision of RAPID results

Optional post-processing of a finished RAPID result (`Morph_final.tif`). RAPID itself
is unchanged; nothing here runs unless called.

## Why

- **Snow.** `S1_download/Snow_detect.py` flags snow if Sentinel-2 SCL saw snow on *any*
  day of a 15-day window, so snow that melted before the S1 date removes correctly
  detected flood (e.g. `S1A_IW_GRDH_1SDV_20220501T001528_..._05229E_121A`, Red River).
- **Desert / beach.** The desert mask (desert raster ∩ ESA WorldCover class 60) removes
  inundated beaches and tidal flats, while RAPID alone over-detects dry beach
  (e.g. `S1A_IW_GRDH_1SDV_20251227T200851_..._07D522_C3E7`, Gulf of Carpentaria).

## Output classes (`<prefix>_Morph_masked.tif`)

| Value | Meaning |
|---|---|
| 0–5 | RAPID classes (0 non-water, 1 permanent water, 2–5 flood) |
| 100 | determined ice: snow on the latest clear S2 view, 1–5 days back (1 = S1 day) |
| 106 | affirmative snow: flood pixel, snow possible, latest clear S2 view shows snow but is 6–15 days old |
| 116 | plausible snow: flood pixel, snow possible, latest clear S2 view shows no snow but is 6–15 days old |
| 200 | undetermined desert: SAR coherence cannot tell inundated from dry |
| 250 | water under S2 cloud / cloud shadow for the whole window: snow undecidable |
| 255 | no data |

`<prefix>_mask_change.tif`: 0 unchanged; desert 1 kept flood, 2 → 0, 3 → 200;
snow 4 → 100, 5 → 106, 6 → 250, 7 → 116; coherent patch 8 → 0; compactness after the
coherent-patch rule 9 → 0.

## Rules

**Desert** (flood classes 2–5 inside the desert mask; Sentinel-1 coherence only):

| Event coherence (pre1, co) | dCoh = baseline (pre2, pre1) − event | Result |
|---|---|---|
| > 0.5 | – | 0 (dry) |
| ≤ 0.5 | ≥ 0.25 | keep flood class |
| ≤ 0.5 | < 0.25, or no coherence | 200 |

Skip coherence when the scene has no desert-mask ∩ flood pixels.

**Snow** (non-ocean water, classes 1–5, where snow is physically possible):

Only a clear S2 view 1–5 days back is affirmative evidence.

| Latest clear S2 view | Result |
|---|---|
| snow, 1–5 days back | 100 |
| no snow, 1–5 days back | flood label kept |
| snow, 6–15 days back | 106 affirmative snow (flood pixels only) |
| no snow, 6–15 days back | 116 plausible snow (flood pixels only; needs band 2 of `snow_latest_clear.py`) |
| none, cloud only (200) | 250, but a desert 200 stays 200 |

106 and 116 need `--snow-possible`: they go only where GFS / ERA5 + DEM say snow or ice
is possible.

Ocean (Natural Earth 10m polygons, rasterized on the RAPID grid) is never changed.

**Coherent patches** (after snow; runs whenever `--coh` is given, i.e. only where
coherence was already fetched for the desert step; `--no-coh-patch` turns it off).
Pixels of flood classes 2–5 are grouped into 8-connected patches; desert-undetermined
200, snow / ice / cloud labels (100, 106, 116, 250), permanent water and non-water split
patches and are never changed. High event coherence means the surface did not change
between pre1 and the event, so the patch is not new water. The rule only deletes and is
strict:

| Condition (all required) | Default option |
|---|---|
| 25th percentile of the patch's event coherence (pre1, co) > 0.75 | `--coh-patch-th 0.75`, `--coh-patch-pct 25` |
| coherence on at least half of the patch's pixels | `--coh-patch-min-cover 0.5` |
| patch at least 0.1 km² (coherence is coarser than the 20 m RAPID grid) | `--coh-patch-min-km2 0.1` |
| → patch set to 0 (change code 7); otherwise unchanged | `--coh-patch-classes 2,3,4,5` |

Patches without enough coherence are kept. The 25th percentile > 0.75 with ≥ 0.1 km²
follows earlier tests (much less deletion of VV-dark, likely real water than median >
0.5); 0.75 fits HyP3 burst coherence (5x1 looks) only.

**Compactness after the coherent-patch rule** (runs right after it; `--no-post-compactness`
turns it off). Deleting patches leaves small fragments, so water is re-checked as whole
objects: 8-connected water of classes 1–5 together, judged with the threshold of its most
reliable class (`--post-compactness 0,1.2,1.6,2.0,2.4`, as RAPID morph). Score = area /
4-connected boundary pixels, as in RAPID morph. Objects containing class 1 are kept; a
failing object loses only its class 2–5 pixels (change code 8).

## Testing status

The snow / ice and desert masks in their current form are tested on two scenes only,
checked against Planet imagery:
`S1A_IW_GRDH_1SDV_20220501T001528_20220501T001603_043010_05229E_121A` (Red River,
frozen lakes next to a valid flood) and
`S1A_IW_GRDH_1SDV_20260107T141552_20260107T141617_062662_07DB2A_B7B0` (California,
desert mask; no snow or ice on water). More snow cases need to be tested before
operational use.

## Inputs and script

The input masks are made by scripts in
[S1_download](https://github.com/WWindrunner/S1_download) (branch `snow-desert-masks`):
`snow_possible.py` (`<S1>_snow_possible.tif`), `snow_latest_clear.py` (`<S1>_ice.tif`,
only when snow is possible) and `find_bursts.py` / `hyp3_coh.py` / `coh_loss.py`
(`<S1>_coh.tif`, only when the desert mask overlaps flood).

The snow step has a real-time and a research product. Within about a week of the S1
time `snow_possible.py` uses GFS (prints `PRODUCT realtime <date>`); run it again from
that date on and it switches to ERA5, whose lake model has ice on more lakes (prints
`PRODUCT research`). Then rerun `snow_latest_clear.py` if it was skipped and the scene
is now `SNOW_POSSIBLE`, and `apply_masks.py`. RAPID itself is not rerun.

`apply_masks.py` (`s1pro` environment) applies the rules above:

```bash
S=S1A_IW_GRDH_1SDV_20251227T200851_20251227T200916_062505_07D522_C3E7
REF=$S/Morph/Morph_final.tif
python apply_masks.py $REF $S --desert $DESERT_MASK_VRT --coh ${S}_coh.tif \
    --snow-possible ${S}_snow_possible.tif        # add --snow ${S}_ice.tif when possible
```

WorldCover is read from `--worldcover-dir` or `RAPID_WORLDCOVER_DIR` when set (as
`AncillaryData.py`), else downloaded. `--desert wc60` uses WorldCover class 60 alone
when the desert raster is unavailable.
