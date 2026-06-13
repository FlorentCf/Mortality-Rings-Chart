# Mortality Rings Visualisation: Dendrochronology Style

Generate tree-ring animations from mortality or other time-series data.

The project renders a dataset as a dendrochronology-inspired cross-section:

- one annual band is one year
- the position around a band is the date within that year
- each cell represents a counted event, such as one death
- local band thickness is driven by the number of cells in that period
- color is driven by mortality versus a baseline, or by another anomaly column you provide
- the animation grows from the center outward, like a tree

The default example uses Belgian weekly deaths from 1992 to 2025.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python generate_mortality_rings.py
```

Default outputs:

```text
outputs/belgium_weekly_deaths_1992_2025.mp4
outputs/belgium_weekly_deaths_1992_2025_h264_aac.mp4
outputs/belgium_weekly_deaths_1992_2025_final_frame.png
outputs/belgium_weekly_deaths_1992_2025_contact_sheet.png
```

The `_h264_aac.mp4` file is encoded with H.264/AVC video, AAC audio, and
`yuv420p` pixels for better social-platform compatibility.

## Belgian Example

The tracked example dataset is:

```text
examples/belgium_weekly_deaths_1992_2025.csv
```

It contains weekly rows with:

```text
year,week_index,start_day,end_day,deaths,baseline,excess
```

In the default render:

- `deaths` controls the number of cells and the local ring thickness
- `excess` controls the color
- `baseline` is included for transparency and reuse

Run the example explicitly:

```powershell
python generate_mortality_rings.py `
  --input examples/belgium_weekly_deaths_1992_2025.csv `
  --name belgium_mortality_rings
```

## Use Another Dataset

You can use either already-binned rows:

```csv
year,start_day,end_day,deaths,baseline,excess
2020,92,98,3500,2400,0.46
```

or date-based rows:

```csv
date,deaths,baseline,excess
2020-04-06,3500,2400,0.46
```

For weekly date rows, the script infers the period length from the date spacing.
You can force it:

```powershell
python generate_mortality_rings.py `
  --input my_weekly_data.csv `
  --date-column week_start `
  --count-column deaths `
  --period-days 7 `
  --name my_rings
```

If your file has no `baseline` or `excess` column, the script estimates a
seasonal rolling baseline from previous years and colors each period against
that expected value.

## Useful Options

```powershell
python generate_mortality_rings.py `
  --input examples/belgium_weekly_deaths_1992_2025.csv `
  --width 1440 `
  --height 1080 `
  --draw-seconds 51 `
  --cell-unit 1 `
  --thickness-gain 0.82 `
  --boundary-smoothing-days 2.2
```

Important knobs:

- `--cell-unit`: counted units per cell. Use `1` for one cell per death.
- `--max-cells`: cap the total cells and let the script scale automatically.
- `--draw-seconds`: how long the tree takes to grow.
- `--width` and `--height`: output resolution. `1440x1080` is 4:3 HD.
- `--palette`: comma-separated hex colors from low to high anomaly.
- `--color-min` and `--color-max`: anomaly values mapped to palette endpoints.
- `--thickness-gain`: exaggerates or softens local thickness differences.
- `--boundary-smoothing-days`: smooths the ring outline across neighboring days.

More details are in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Repository Shape

```text
generate_mortality_rings.py                  reusable animation generator
examples/belgium_weekly_deaths_1992_2025.csv Belgian example input
docs/DATA_FORMAT.md                          input format and mapping notes
requirements.txt                             Python dependencies
```

Generated videos, preview frames, downloaded source files, and scratch outputs
are written to `outputs/` and ignored by git.

## Data Source

Belgian mortality example:
Statbel open data, daily deaths aggregated into weekly periods for 1992-2025.
