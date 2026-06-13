# HYDE Mapping Notes

HYDE 3.5 is very different from the Belgium mortality data. It is not weekly or
monthly event data. It is a long historical land-use reconstruction, with
irregular historical timesteps from 10,000 BCE to 2025 and multiple land-use
signals.

## Implemented Prototype

`generate_hyde_land_use_rings.py` uses the official HYDE 3.5 base-scenario
historical CSV files:

- `his_crop_4apr2025.csv`
- `his_past_4apr2025.csv`
- `his_rice_4apr2025.csv`
- `his_irri_4apr2025.csv`

Current visual grammar:

- ring = HYDE historical timestep
- angle = land-use signal
- cell count = scaled signal area
- color = land-use signal
- ring thickness = primary land-use footprint, cropland plus pasture

This avoids pretending that HYDE has monthly or weekly structure. Time moves
radially outward; composition lives around the 360 degrees.

## Why Cells Are Scaled

The HYDE values are land-area totals, not discrete events like deaths. Rendering
one cell per square kilometer would be both too dense and misleading. The script
therefore calculates a dynamic `cell_unit` so the largest land-use signal creates
a readable number of cells while preserving relative size.

## Alternative Models Worth Testing

1. **Region-around-the-ring**
   - Ring = year/timestep
   - Angle = world region or country
   - Cell count = cropland/pasture/population
   - Color = land-use category

   This could show the geographic spread of human land use more poetically than
   the current category-sector model.

2. **Category-as-rings**
   - Ring = land-use category
   - Angle = historical time
   - Cell count = area change
   - Color = era or growth rate

   This is less tree-like but may make acceleration after 1700 and 1950 easier
   to read.

3. **Change-only cambium**
   - Ring = timestep
   - Angle = region or category
   - Cells = positive change since previous timestep
   - Color = expansion versus contraction

   This would reduce the dominance of present-day land-use stock and make the
   historical pulses more visible.

## Current Recommendation

Use the implemented prototype as a technical proof that the dendrochronology
engine can ingest HYDE. For a final editorial animation, the strongest next
variant is probably **region-around-the-ring**, because HYDE's real drama is the
geographic spread of land transformation over millennia.
