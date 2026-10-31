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
| 106–115 | less credible ice: 100 + days back (6–15) |
| 200 | undetermined desert: SAR coherence cannot tell inundated from dry |
| 250 | water under S2 cloud / cloud shadow for the whole window: snow undecidable |
| 255 | no data |

`<prefix>_mask_change.tif`: 0 unchanged; desert 1 kept flood, 2 → 0, 3 → 200;
snow 4 → 100, 5 → 106–115, 6 → 250.

## Rules

**Desert** (flood classes 2–5 inside the desert mask; Sentinel-1 coherence only):

| Event coherence (pre1, co) | dCoh = baseline (pre2, pre1) − event | Result |
|---|---|---|
| > 0.5 | – | 0 (dry) |
| ≤ 0.5 | ≥ 0.25 | keep flood class |
| ≤ 0.5 | < 0.25, or no coherence | 200 |

Skip coherence when the scene has no desert-mask ∩ flood pixels.

**Snow** (non-ocean water, classes 1–5, where snow is physically possible):

| Snow mask | Result |
|---|---|
| 1–5 | 100 |
| 6–15 | 100 + days |
| 200 (cloud only) | 250, but a desert 200 stays 200 |

Ocean (Natural Earth 10m polygons, rasterized on the RAPID grid) is never changed.

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
