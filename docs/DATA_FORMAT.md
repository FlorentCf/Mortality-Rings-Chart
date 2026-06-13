# Data Format

`generate_mortality_rings.py` accepts CSV input. The visual model is general,
but the default column names are mortality-oriented.

## Option 1: Already-Binned Data

Use this when you already know the year and position within each year.

```csv
year,start_day,end_day,deaths,baseline,excess
2020,92,98,3500,2400,0.46
2020,99,105,4100,2450,0.67
```

Default day indexing is one-based, so `start_day=1` means January 1. Use
`--day-indexing zero-based` if your days run from `0` to `364` or `365`.

Required columns:

- `year`
- `start_day`
- `end_day`
- a count column, default `deaths`

Optional columns:

- `baseline`: expected count for the same period
- `excess`: anomaly used for color, usually `count / baseline - 1`

If `excess` is present, the script uses it directly for color.

## Option 2: Date-Based Data

Use this for daily, weekly, or regular time-series files.

```csv
date,deaths,baseline,excess
2020-04-06,3500,2400,0.46
2020-04-13,4100,2450,0.67
```

The script infers the period length from the spacing between dates. For weekly
data, you can be explicit:

```powershell
python generate_mortality_rings.py --input data.csv --period-days 7
```

If a period crosses December/January, the script splits it across both years so
the annual rings stay calendar-year based.

## Column Mapping

Use CLI flags when your file has different names:

```powershell
python generate_mortality_rings.py `
  --input data.csv `
  --date-column week_start `
  --count-column value `
  --baseline-column expected `
  --excess-column anomaly
```

## Visual Mapping

The default mapping is:

```text
ring        = year
angle       = date within year
cells       = count / cell_unit
thickness   = local count intensity inside that year
color       = excess or estimated anomaly versus baseline
animation   = radial growth from inner edge to outer edge
```

For non-mortality data, rename the count mentally:

- `deaths` can be any non-negative event count
- `baseline` can be expected traffic, expected claims, expected rainfall, etc.
- `excess` can be any signed anomaly you want the palette to encode

## Baseline Behavior

If `excess` is missing but `baseline` exists, the script computes:

```text
excess = count / baseline - 1
```

If both `baseline` and `excess` are missing, it estimates a seasonal baseline
from the same week in nearby previous years. This is useful for previews, but a
domain-specific baseline is better for final editorial work.

## Cell Scaling

By default:

```text
1 cell = 1 counted unit
```

For very large datasets:

```powershell
python generate_mortality_rings.py --input data.csv --max-cells 3000000
```

The script will increase `cell-unit` automatically and report the scale.

## Palette

The default palette expects anomaly values roughly from `-50%` to `+120%`:

```text
#1B263B,#2F5266,#6F8F91,#B2B8AA,#E7DDC8,#D6A84F,#B7653A,#872B36,#4A1022,#1F0610
```

Use another palette:

```powershell
python generate_mortality_rings.py `
  --palette "#fff5f0,#fee0d2,#fcbba1,#fc9272,#fb6a4a,#ef3b2c,#cb181d,#a50f15,#67000d" `
  --color-min 0 `
  --color-max 1
```

## Output Files

For `--name my_rings`, the script writes:

```text
outputs/my_rings.mp4
outputs/my_rings_h264_aac.mp4
outputs/my_rings_final_frame.png
outputs/my_rings_contact_sheet.png
```
