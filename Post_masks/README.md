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

## Scripts

| Script | Environment | Purpose |
|---|---|---|
| `snow_possible.py` | `s1pro` + xarray, zarr≥3, gcsfs | 1 = snow possible: 30-day minimum 2 m temperature (NOAA GFS 0.25° TMIN, a few hours latency; ERA5 for windows before 2021), lapse-rate (6.5 K/km) downscaled to Copernicus DEM GLO-90, ≤ 5 °C. Prints `SNOW_POSSIBLE` / `SNOW_IMPOSSIBLE`. |
| `snow_latest_clear.py` | `s1pro` | S2 snow mask from the latest clear SCL view in the 15-day window (values above). Only needed when snow is possible. |
| `find_bursts.py` | asf_search | SLC bursts of the GRD scene and their two previous acquisitions. |
| `hyp3_coh.py` | hyp3_sdk | Submit / fetch ASF HyP3 `INSAR_ISCE_BURST` jobs (looks 5x1, 1 credit per pair) for event and baseline pairs. |
| `coh_loss.py` | `s1pro` | Mosaic burst coherence onto the RAPID grid: band 1 baseline, 2 event, 3 loss. |
| `apply_masks.py` | `s1pro` | Apply the rules above. |

## Example

```bash
S=S1A_IW_GRDH_1SDV_20251227T200851_20251227T200916_062505_07D522_C3E7
REF=$S/Morph/Morph_final.tif

# Snow: physical gate first, the S2 mask only if needed
python snow_possible.py $S $REF ${S}_snow_possible.tif | tail -1     # SNOW_IMPOSSIBLE
# python snow_latest_clear.py $S $REF ${S}_ice.tif

# Desert: coherence only if desert mask ∩ flood is not empty
python find_bursts.py $S bursts.json
python hyp3_coh.py submit bursts.json coh_$S      # later: fetch
python coh_loss.py coh_$S $REF $S                 # -> ${S}_coh.tif

python apply_masks.py $REF $S --desert $DESERT_MASK_VRT --coh ${S}_coh.tif \
    --snow-possible ${S}_snow_possible.tif        # add --snow ${S}_ice.tif when possible
```

`--desert wc60` uses WorldCover class 60 alone when the desert raster is unavailable.
HyP3 needs NASA Earthdata credentials (`~/.netrc`).
