# Dendrochronology Viz

Generative visualizations inspired by dendrochronology: time, change, and
composition rendered as tree-ring growth.

The default project is **Belgium Mortality Rings**, a weekly mortality animation
where the chart grows from the center outward like a tree cross-section.

## Default: Belgium Mortality Rings

The flagship render uses Belgian daily deaths from Statbel, aggregated into
weekly signals for 1992-2025.

Visual contract:

- one annual band = one year
- one plotted cell = one death
- position around a band = week of year
- local thickness = number of deaths
- color = mortality compared with the expected baseline
- growth = radial, from the inner edge of each year to the outer edge

Run the share-ready 4:3 version:

```powershell
python generate_belgium_mortality_rings_final.py
```

Main output:

```text
outputs/belgium_mortality_rings_final_4x3_1992_2025_x_h264_aac.mp4
```

The `_x_h264_aac.mp4` output is encoded with H.264/AVC video, AAC audio, and
`yuv420p` pixels for social-platform compatibility.

## Belgium Variants

Chart-only render, without title or labels:

```powershell
python generate_belgium_mortality_rings_chart_only.py
```

Context-color experiment, where thickness stays death-count-only and color uses
a separate weather/respiratory/cause-of-death context score:

```powershell
python generate_belgium_mortality_rings_context_color.py
```

Legacy 3-seconds-per-year Belgium render:

```powershell
python generate_belgium_weekly_mortality_tree.py
```

## HYDE Land-Use Rings

The HYDE experiment uses the official HYDE 3.5 public package from Utrecht
University / PBL.

Instead of forcing time around the circle, this version maps:

- ring = historical timestep, from 10,000 BCE to 2025
- angle = land-use signal
- cells = scaled land-use area signal
- color = land-use signal

Signals currently used:

- cropland
- pasture
- rice
- irrigation

Run it:

```powershell
python generate_hyde_land_use_rings.py
```

Main output:

```text
outputs/hyde_land_use_rings/hyde35_land_use_rings_10000BCE_2025_x_h264_aac.mp4
```

This is a prototype model, not the final editorial form. It is intentionally
kept separate from the default Belgium mortality animation.

## Eurostat Country Examples

Five non-Belgium examples can be generated from Eurostat weekly deaths:

```powershell
python generate_eurostat_country_mortality_rings.py
```

The script renders Netherlands, Sweden, Finland, Switzerland, and Austria with
separate palettes.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Data Sources

- Belgium mortality: Statbel open data, `TF_DEATHS`, number of deaths per day.
- Eurostat mortality examples: `demo_r_mwk_ts`, deaths by week and sex.
- HYDE land use: Utrecht University / PBL, HYDE 3.5 public data package,
  DOI `10.24416/UU01-F45D44`.

Generated videos, source downloads, CSV extracts, and preview images are written
to `outputs/` and are intentionally ignored by git.

## Repository Shape

```text
generate_belgium_mortality_rings_final.py      default share-ready render
generate_belgium_mortality_rings_chart_only.py chart-only Belgium render
generate_belgium_mortality_rings_context_color.py
                                               context-color Belgium variant
generate_belgium_weekly_mortality_tree.py      legacy Belgium data/model base
generate_eurostat_country_mortality_rings.py   country examples
generate_hyde_land_use_rings.py                HYDE land-use experiment
requirements.txt
```
