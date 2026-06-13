# Mortality Rings Visualisation: Dendrochronology Style

Generate dendrochronology-inspired mortality animations: a tree cross-section
where years grow outward, dates sit around each ring, and cells represent
deaths.

The canonical render in this repository is the **Belgium chart-only mortality
rings** animation for 1992-2025.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python generate_belgium_mortality_rings_chart_only.py
```

Main outputs:

```text
outputs/belgium_mortality_rings_chart_only_4x3_1992_2025.mp4
outputs/belgium_mortality_rings_chart_only_4x3_1992_2025_x_h264_aac.mp4
outputs/belgium_mortality_rings_chart_only_4x3_1992_2025_final_frame.png
outputs/belgium_mortality_rings_chart_only_4x3_1992_2025_contact_sheet.png
```

The `_x_h264_aac.mp4` file is encoded with H.264/AVC video, AAC audio, and
`yuv420p` pixels for better social-platform compatibility.

## Visual Grammar

- one annual band = one year
- one cell = one death
- position around a band = date within the year
- band thickness = local death volume
- color = mortality versus expected baseline
- animation = radial growth from the center outward

The default video is intentionally chart-only: no title, no labels, no legend,
just the tree rings growing on a white background.

## Belgian Example Data

The repo includes the Belgian example data needed by the canonical renderer:

```text
examples/belgium_daily_deaths_1992_2025.csv
examples/belgium_weekly_deaths_1992_2025.csv
```

The daily file is used by the final renderer because cells are placed day by
day. The weekly file is included as a compact analysis-ready example with
precomputed baseline and excess values.

Belgian source: Statbel open data, number of deaths per day.

## Other Renderers

Annotated version with title and labels:

```powershell
python generate_belgium_mortality_rings_final.py
```

Generic CSV-driven renderer for adapting the method to another dataset:

```powershell
python generate_mortality_rings.py --input examples/belgium_weekly_deaths_1992_2025.csv
```

The generic renderer is useful for experimentation, but it is not a pixel match
for the canonical Belgian chart-only render.

## Use Another Dataset

For reusable data input, see [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

The generic renderer accepts already-binned rows:

```csv
year,start_day,end_day,deaths,baseline,excess
2020,92,98,3500,2400,0.46
```

or date-based rows:

```csv
date,deaths,baseline,excess
2020-04-06,3500,2400,0.46
```

If no baseline is provided, it estimates a seasonal rolling baseline from
nearby previous years. For final editorial work, prefer a documented
domain-specific baseline.

## Repository Shape

```text
generate_belgium_mortality_rings_chart_only.py canonical chart-only render
generate_belgium_mortality_rings_final.py      annotated Belgian render
generate_belgium_weekly_mortality_tree.py      shared Belgian data/model code
generate_mortality_rings.py                    generic CSV renderer
examples/                                      Belgian example data
docs/DATA_FORMAT.md                            reusable input notes
requirements.txt                               Python dependencies
```

Generated videos, preview frames, downloaded source files, and scratch outputs
are written to `outputs/` and ignored by git.
