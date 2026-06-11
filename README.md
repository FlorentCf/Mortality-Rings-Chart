# Dendrochronology Viz

Generative animation experiments that map tree-ring growth ideas onto mortality
time series. The project started as a synthetic tree-growth animation and now
includes data-driven renderers for Swiss monthly mortality and Belgian weekly
mortality.

The latest renderer creates a Belgium mortality-ring animation where:

- one annual band represents one year,
- one plotted cell represents one death,
- angle within the ring represents time within the year,
- weekly excess mortality drives color,
- weekly death counts and excess mortality drive local ring thickness,
- each year grows radially from the inner edge to the outer edge,
- the Dec-Jan seam is blended as a circular year boundary.

Generated videos, source downloads, CSV extracts, and preview images are written
to `outputs/` and are intentionally ignored by git.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Main Render

Belgium weekly mortality, 1992-2025, 3 seconds per year:

```powershell
python generate_belgium_weekly_mortality_tree.py
```

This downloads Statbel daily deaths, aggregates them into calendar-week signals,
then renders:

- `outputs/belgium_weekly_mortality_tree_1992_2025_radial_growth_3s_per_year.mp4`
- `outputs/belgium_weekly_mortality_tree_1992_2025_radial_growth_3s_per_year_x_h264_aac.mp4`
- final-frame and contact-sheet PNG previews

The `_x_h264_aac.mp4` file is re-encoded with H.264 video, AAC audio, and
`yuv420p` pixels for social-platform compatibility.

## Other Renderers

```powershell
python generate_tree_growth.py
```

Synthetic tree-cell growth animation.

```powershell
python generate_mortality_tree.py
python generate_mortality_tree_one_death.py
```

Swiss mortality-ring experiments based on the Swiss BFS PXWEB table. These
scripts expect the BFS query JSON used during development or can be adapted to a
direct query payload.

```powershell
python make_web_versions.py
python make_x_compatible.py
```

Helpers for downscaling and re-encoding generated MP4 files.

## Data Sources

- Belgium: Statbel open data, `TF_DEATHS`, number of deaths per day.
- Switzerland: Swiss Federal Statistical Office PXWEB table for deaths per
  month and mortality since 1803.

## Requirements

- Python 3.10+
- `numpy`
- `Pillow`
- `opencv-python`
- `imageio-ffmpeg`
- `pandas`
- `openpyxl`
