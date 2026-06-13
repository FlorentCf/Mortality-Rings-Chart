from __future__ import annotations

import argparse
import calendar
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_PALETTE = [
    "#1B263B",
    "#2F5266",
    "#6F8F91",
    "#B2B8AA",
    "#E7DDC8",
    "#D6A84F",
    "#B7653A",
    "#872B36",
    "#4A1022",
    "#1F0610",
]

ANGLE_SAMPLES = 1440
WEEK_COUNT = 52
ANGLE_OFFSET = -math.pi / 2.0 + math.radians(17.0)
BACKGROUND = (255, 255, 255)
DEFAULT_NEUTRAL = "#E7DDC8"


@dataclass(slots=True)
class RawBin:
    year: int
    start_day: int
    end_day: int
    count: float
    baseline: float | None
    excess: float | None


@dataclass(slots=True)
class RingBin:
    year_index: int
    year: int
    start_day: int
    end_day: int
    count: float
    cells: int
    baseline: float
    excess: float
    days: int


@dataclass(slots=True)
class RingModel:
    years: list[int]
    bins_by_year: list[list[RingBin]]
    color_signals: list[np.ndarray]
    boundaries: list[np.ndarray]
    total_count: float
    total_cells: int
    cell_unit: float


@dataclass(slots=True)
class RectChunk:
    x: np.ndarray
    y: np.ndarray
    rgb: np.ndarray
    tx: np.ndarray
    ty: np.ndarray


@dataclass(slots=True)
class RenderConfig:
    width: int
    height: int
    fps: int
    draw_seconds: float
    hold_seconds: float
    center_x: float
    center_y: float
    pith_radius: float
    max_radius: float
    seed: int
    neutral_rgb: tuple[int, int, int]
    cell_unit: float
    max_cells: int
    thickness_gain: float
    thickness_response: float
    min_local_width: float
    max_local_width: float
    boundary_smoothing_days: float
    color_smoothing_days: float
    angle_jitter_days: float
    radial_jitter: float
    color_jitter: float
    output_dir: Path
    name: str
    encode_h264: bool
    save_contact_sheet: bool


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    clean = value.strip().lstrip("#")
    if len(clean) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {value!r}")
    return tuple(int(clean[index : index + 2], 16) for index in (0, 2, 4))


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def circular_gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.astype(float, copy=True)
    radius = max(2, int(math.ceil(sigma * 4.0)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    result = np.zeros_like(values, dtype=float)
    for weight, offset in zip(kernel, offsets):
        result += weight * np.roll(values, int(offset))
    return result


def circular_interp(values: np.ndarray, position: np.ndarray) -> np.ndarray:
    wrapped = np.asarray(position) % len(values)
    left = np.floor(wrapped).astype(int) % len(values)
    right = (left + 1) % len(values)
    frac = wrapped - np.floor(wrapped)
    return values[left] * (1.0 - frac) + values[right] * frac


def sample_periodic(values: np.ndarray, angle: np.ndarray | float) -> np.ndarray | float:
    position = (np.asarray(angle) % math.tau) / math.tau * len(values)
    sampled = circular_interp(values, position)
    if np.isscalar(angle):
        return float(sampled)
    return sampled


def closed(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points + points[:1]


def polar(config: RenderConfig, radius: float, angle: float) -> tuple[float, float]:
    visual_angle = angle + ANGLE_OFFSET
    return (
        config.center_x + radius * math.cos(visual_angle),
        config.center_y + radius * math.sin(visual_angle),
    )


def parse_palette(colors: str, color_min: float, color_max: float) -> tuple[np.ndarray, np.ndarray]:
    hex_values = [item.strip() for item in colors.split(",") if item.strip()]
    if len(hex_values) < 2:
        raise ValueError("Palette must contain at least two comma-separated hex colors.")
    stops = np.linspace(color_min, color_max, len(hex_values), dtype=float)
    palette = np.array([hex_to_rgb(value) for value in hex_values], dtype=float)
    return stops, palette


def colors_for_metric(values: np.ndarray, stops: np.ndarray, palette: np.ndarray) -> np.ndarray:
    clamped = np.clip(values, stops[0], stops[-1])
    red = np.interp(clamped, stops, palette[:, 0])
    green = np.interp(clamped, stops, palette[:, 1])
    blue = np.interp(clamped, stops, palette[:, 2])
    return np.column_stack((red, green, blue)).astype(np.float32)


def pick_column(df: pd.DataFrame, requested: str | None, candidates: list[str]) -> str | None:
    columns = {column.lower(): column for column in df.columns}
    if requested and requested in df.columns:
        return requested
    if requested and requested.lower() in columns:
        return columns[requested.lower()]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    return None


def infer_period_days(dates: pd.Series) -> int:
    unique_dates = pd.Series(sorted(pd.to_datetime(dates).dropna().unique()))
    if len(unique_dates) <= 1:
        return 1
    diffs = unique_dates.diff().dropna().dt.days
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1
    return max(1, int(round(float(diffs.median()))))


def add_date_row_as_bins(
    rows: list[RawBin],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    count: float,
    baseline: float | None,
    excess: float | None,
) -> None:
    total_days = max(1, (end_date - start_date).days + 1)
    current = start_date
    while current <= end_date:
        year_end = pd.Timestamp(year=current.year, month=12, day=31)
        segment_end = min(end_date, year_end)
        segment_days = (segment_end - current).days + 1
        fraction = segment_days / total_days
        scaled_baseline = baseline * fraction if baseline is not None else None
        rows.append(
            RawBin(
                year=int(current.year),
                start_day=int(current.dayofyear) - 1,
                end_day=int(segment_end.dayofyear) - 1,
                count=float(count) * fraction,
                baseline=scaled_baseline,
                excess=excess,
            )
        )
        current = segment_end + pd.Timedelta(days=1)


def load_raw_bins(args: argparse.Namespace) -> list[RawBin]:
    df = pd.read_csv(args.input)
    df.columns = [str(column).strip() for column in df.columns]

    count_column = pick_column(df, args.count_column, ["deaths", "count", "value", "cells"])
    if not count_column:
        raise ValueError("Could not find a count column. Use --count-column to specify it.")

    baseline_column = pick_column(df, args.baseline_column, ["baseline", "expected"])
    excess_column = pick_column(df, args.excess_column, ["excess", "anomaly", "ratio_vs_baseline"])

    year_column = pick_column(df, args.year_column, ["year"])
    start_day_column = pick_column(df, args.start_day_column, ["start_day", "day_start"])
    end_day_column = pick_column(df, args.end_day_column, ["end_day", "day_end"])

    rows: list[RawBin] = []
    if year_column and start_day_column and end_day_column:
        offset = 1 if args.day_indexing == "one-based" else 0
        for _index, row in df.iterrows():
            count = float(row[count_column])
            if count <= 0 or not math.isfinite(count):
                continue
            baseline = float(row[baseline_column]) if baseline_column else None
            excess = float(row[excess_column]) if excess_column else None
            rows.append(
                RawBin(
                    year=int(row[year_column]),
                    start_day=int(row[start_day_column]) - offset,
                    end_day=int(row[end_day_column]) - offset,
                    count=count,
                    baseline=baseline,
                    excess=excess,
                )
            )
        return rows

    date_column = pick_column(df, args.date_column, ["date", "period_start", "start_date"])
    if not date_column:
        raise ValueError(
            "Input must contain either year/start_day/end_day columns or a date column. "
            "Use --date-column if needed."
        )

    end_date_column = pick_column(df, args.end_date_column, ["end_date", "period_end"])
    dates = pd.to_datetime(df[date_column])
    period_days = args.period_days or infer_period_days(dates)
    for index, row in df.iterrows():
        count = float(row[count_column])
        if count <= 0 or not math.isfinite(count):
            continue
        start_date = pd.Timestamp(row[date_column])
        if end_date_column:
            end_date = pd.Timestamp(row[end_date_column])
        else:
            end_date = start_date + pd.Timedelta(days=period_days - 1)
        baseline = float(row[baseline_column]) if baseline_column else None
        excess = float(row[excess_column]) if excess_column else None
        add_date_row_as_bins(rows, start_date, end_date, count, baseline, excess)
    return rows


def days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def week_index_for_day(day: int, year_days: int) -> int:
    return min(WEEK_COUNT - 1, int(day / year_days * WEEK_COUNT))


def compute_missing_baselines(rows: list[RawBin], years: list[int], window: int) -> dict[tuple[int, int], float]:
    year_to_index = {year: index for index, year in enumerate(years)}
    sums = np.zeros((len(years), WEEK_COUNT), dtype=float)
    days = np.zeros((len(years), WEEK_COUNT), dtype=float)
    for item in rows:
        year_days = days_in_year(item.year)
        mid_day = (item.start_day + item.end_day) / 2.0
        week = week_index_for_day(int(mid_day), year_days)
        y = year_to_index[item.year]
        item_days = max(1, item.end_day - item.start_day + 1)
        sums[y, week] += item.count
        days[y, week] += item_days

    rates = np.divide(sums, np.maximum(1.0, days), out=np.zeros_like(sums), where=days > 0)
    baselines = np.zeros_like(rates)
    for year_index in range(len(years)):
        for week in range(WEEK_COUNT):
            start = max(0, year_index - window)
            previous = rates[start:year_index, week]
            previous = previous[previous > 0]
            if len(previous) >= 3:
                baseline = float(previous.mean())
            else:
                peers = rates[:, week]
                peers = peers[peers > 0]
                baseline = float(peers.mean()) if len(peers) else 1.0
            baselines[year_index, week] = max(1e-9, baseline)
    for year_index in range(len(years)):
        baselines[year_index] = circular_gaussian(baselines[year_index], sigma=0.80)

    result: dict[tuple[int, int], float] = {}
    for year_index, year in enumerate(years):
        for week in range(WEEK_COUNT):
            result[(year, week)] = float(baselines[year_index, week])
    return result


def prepare_bins(raw_rows: list[RawBin], args: argparse.Namespace, cell_unit: float) -> tuple[list[int], list[list[RingBin]]]:
    filtered = [
        row
        for row in raw_rows
        if (args.start_year is None or row.year >= args.start_year)
        and (args.end_year is None or row.year <= args.end_year)
    ]
    if not filtered:
        raise ValueError("No rows left after applying the year filters.")

    years = sorted({row.year for row in filtered})
    year_to_index = {year: index for index, year in enumerate(years)}
    computed_baselines = compute_missing_baselines(filtered, years, args.baseline_window)
    bins_by_year: list[list[RingBin]] = [[] for _ in years]

    for row in filtered:
        year_days = days_in_year(row.year)
        start_day = max(0, min(year_days - 1, int(row.start_day)))
        end_day = max(start_day, min(year_days - 1, int(row.end_day)))
        item_days = max(1, end_day - start_day + 1)
        mid_day = (start_day + end_day) / 2.0
        week = week_index_for_day(int(mid_day), year_days)
        rate = row.count / item_days

        if row.excess is not None and math.isfinite(row.excess):
            excess = float(row.excess)
            baseline = float(row.baseline) if row.baseline is not None else rate / max(1e-9, 1.0 + excess)
        elif row.baseline is not None and row.baseline > 0:
            baseline = float(row.baseline)
            excess = row.count / baseline - 1.0
        else:
            baseline_rate = computed_baselines[(row.year, week)]
            baseline = baseline_rate * item_days
            excess = rate / baseline_rate - 1.0

        cells = int(round(row.count / cell_unit))
        if row.count > 0 and cells == 0:
            cells = 1
        bins_by_year[year_to_index[row.year]].append(
            RingBin(
                year_index=year_to_index[row.year],
                year=row.year,
                start_day=start_day,
                end_day=end_day,
                count=float(row.count),
                cells=cells,
                baseline=float(baseline),
                excess=float(excess),
                days=item_days,
            )
        )

    for year_bins in bins_by_year:
        year_bins.sort(key=lambda item: (item.start_day, item.end_day))
    return years, bins_by_year


def build_color_signals(years: list[int], bins_by_year: list[list[RingBin]], config: RenderConfig) -> list[np.ndarray]:
    signals: list[np.ndarray] = []
    for year, bins in zip(years, bins_by_year):
        year_days = days_in_year(year)
        daily = np.zeros(year_days, dtype=float)
        weights = np.zeros(year_days, dtype=float)
        for item in bins:
            daily[item.start_day : item.end_day + 1] += item.excess
            weights[item.start_day : item.end_day + 1] += 1.0
        mask = weights > 0
        daily[mask] /= weights[mask]
        daily = circular_gaussian(daily, sigma=config.color_smoothing_days)
        samples = np.zeros(ANGLE_SAMPLES, dtype=float)
        for sample_index in range(ANGLE_SAMPLES):
            day_position = sample_index / ANGLE_SAMPLES * year_days
            left = int(math.floor(day_position)) % year_days
            right = (left + 1) % year_days
            frac = day_position - math.floor(day_position)
            samples[sample_index] = daily[left] * (1.0 - frac) + daily[right] * frac
        signals.append(circular_gaussian(samples, sigma=config.color_smoothing_days * ANGLE_SAMPLES / year_days))
    return signals


def build_boundaries(years: list[int], bins_by_year: list[list[RingBin]], config: RenderConfig) -> list[np.ndarray]:
    annual_totals = np.array([sum(item.count for item in bins) for bins in bins_by_year], dtype=float)
    positive_mean = float(annual_totals[annual_totals > 0].mean()) if np.any(annual_totals > 0) else 1.0
    mean_widths = (np.maximum(annual_totals, 1.0) / positive_mean) ** 0.35
    mean_widths = mean_widths / mean_widths.sum() * (config.max_radius - config.pith_radius)

    boundaries = [np.full(ANGLE_SAMPLES, config.pith_radius, dtype=float)]
    for year, bins, mean_width in zip(years, bins_by_year, mean_widths):
        year_days = days_in_year(year)
        annual_rate = sum(item.count for item in bins) / max(1, year_days)
        local = np.ones(year_days, dtype=float)
        for item in bins:
            rate = item.count / max(1, item.days)
            relative = rate / max(1e-9, annual_rate) - 1.0
            factor = 1.0 + config.thickness_gain * math.tanh(relative / max(1e-6, config.thickness_response))
            local[item.start_day : item.end_day + 1] = factor

        local = circular_gaussian(local, sigma=config.boundary_smoothing_days)
        local = np.clip(local, config.min_local_width, config.max_local_width)
        local /= max(1e-9, float(local.mean()))

        samples = np.zeros(ANGLE_SAMPLES, dtype=float)
        for sample_index in range(ANGLE_SAMPLES):
            day_position = sample_index / ANGLE_SAMPLES * year_days
            left = int(math.floor(day_position)) % year_days
            right = (left + 1) % year_days
            frac = day_position - math.floor(day_position)
            samples[sample_index] = local[left] * (1.0 - frac) + local[right] * frac
        samples = circular_gaussian(samples, sigma=config.boundary_smoothing_days * ANGLE_SAMPLES / year_days)
        samples /= max(1e-9, float(samples.mean()))
        boundaries.append(boundaries[-1] + mean_width * samples)
    return boundaries


def build_model(args: argparse.Namespace, config: RenderConfig) -> RingModel:
    raw_bins = load_raw_bins(args)
    filtered_raw_bins = [
        item
        for item in raw_bins
        if (args.start_year is None or item.year >= args.start_year)
        and (args.end_year is None or item.year <= args.end_year)
    ]
    total_count = float(sum(item.count for item in filtered_raw_bins))
    cell_unit = max(1e-9, config.cell_unit)
    if config.max_cells > 0 and total_count / cell_unit > config.max_cells:
        cell_unit = total_count / config.max_cells
        print(f"auto cell scale: 1 cell = {cell_unit:,.2f} units")

    years, bins_by_year = prepare_bins(filtered_raw_bins, args, cell_unit)
    color_signals = build_color_signals(years, bins_by_year, config)
    boundaries = build_boundaries(years, bins_by_year, config)
    total_cells = sum(item.cells for bins in bins_by_year for item in bins)
    return RingModel(
        years=years,
        bins_by_year=bins_by_year,
        color_signals=color_signals,
        boundaries=boundaries,
        total_count=total_count,
        total_cells=total_cells,
        cell_unit=cell_unit,
    )


def radius_at(model: RingModel, year_index: int, radial_t: np.ndarray, angle: np.ndarray) -> np.ndarray:
    inner = sample_periodic(model.boundaries[year_index], angle)
    outer = sample_periodic(model.boundaries[year_index + 1], angle)
    return inner + (outer - inner) * radial_t


def boundary_points(model: RingModel, config: RenderConfig, boundary_index: int, samples: int = 720) -> list[tuple[float, float]]:
    values = model.boundaries[boundary_index]
    phase = (boundary_index * 2.399963229728653) % math.tau
    return [
        polar(config, float(sample_periodic(values, phase + i * math.tau / samples)), phase + i * math.tau / samples)
        for i in range(samples)
    ]


def oriented_pixel_offsets(visual_angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tx = np.rint(-np.sin(visual_angle)).astype(np.int8)
    ty = np.rint(np.cos(visual_angle)).astype(np.int8)
    tx[(tx == 0) & (ty == 0)] = 1
    return tx, ty


def generate_frame_chunks(
    model: RingModel,
    config: RenderConfig,
    color_stops: np.ndarray,
    palette: np.ndarray,
) -> tuple[list[list[RectChunk]], np.ndarray]:
    frame_count = max(1, int(round(config.draw_seconds * config.fps)))
    chunks: list[list[RectChunk]] = [[] for _ in range(frame_count)]
    frame_counts = np.zeros(frame_count, dtype=np.int64)
    year_count = len(model.years)

    for year_index, (year, bins) in enumerate(zip(model.years, model.bins_by_year)):
        year_days = days_in_year(year)
        color_signal = model.color_signals[year_index]
        for bin_index, item in enumerate(bins):
            n = item.cells
            if n <= 0:
                continue
            rng = np.random.default_rng(config.seed + year * 10_000 + item.start_day * 101 + bin_index)
            day_position = item.start_day + rng.random(n) * max(1, item.days)
            theta = math.tau * (day_position / year_days)
            theta += rng.normal(0.0, math.tau * config.angle_jitter_days / year_days, n)

            radial_t = rng.random(n)
            theta += (radial_t - 0.5) * rng.normal(0.0, math.tau * 0.42 / year_days, n)

            radius = radius_at(model, year_index, radial_t, theta)
            radius += rng.normal(0.0, config.radial_jitter, n)
            visual_angle = theta + ANGLE_OFFSET
            x = np.rint(config.center_x + radius * np.cos(visual_angle) + rng.normal(0.0, 0.16, n)).astype(np.int32)
            y = np.rint(config.center_y + radius * np.sin(visual_angle) + rng.normal(0.0, 0.16, n)).astype(np.int32)

            color_position = (theta % math.tau) / math.tau * ANGLE_SAMPLES
            metric = circular_interp(color_signal, color_position)
            base_color = colors_for_metric(metric, color_stops, palette)
            rgb = np.clip(base_color + rng.normal(0.0, config.color_jitter, (n, 3)), 0, 255).astype(np.uint8)
            tx, ty = oriented_pixel_offsets(visual_angle)

            radial_progress = np.clip(radial_t + rng.normal(0.0, 0.014, n), 0.0, 0.998)
            progress = (year_index + radial_progress) / year_count
            birth_frames = np.clip((progress * (frame_count - 1)).astype(np.int32), 0, frame_count - 1)
            for frame_index in np.unique(birth_frames):
                mask = birth_frames == frame_index
                chunks[int(frame_index)].append(
                    RectChunk(x=x[mask], y=y[mask], rgb=rgb[mask], tx=tx[mask], ty=ty[mask])
                )
                frame_counts[int(frame_index)] += int(mask.sum())

        if year_index % 5 == 0 or year_index == year_count - 1:
            print(f"prepared cells through {year}")
    return chunks, frame_counts


def plot_rect_cells(canvas: np.ndarray, chunk: RectChunk, config: RenderConfig) -> None:
    x = chunk.x
    y = chunk.y
    valid = (x >= 1) & (x < config.width - 2) & (y >= 1) & (y < config.height - 2)
    if not np.any(valid):
        return
    xv = x[valid]
    yv = y[valid]
    rgb = chunk.rgb[valid]
    tx = chunk.tx[valid]
    ty = chunk.ty[valid]
    canvas[yv, xv] = rgb
    canvas[yv + ty, xv + tx] = rgb


def current_year_index(model: RingModel, frame_index: int, draw_frame_count: int) -> float:
    if frame_index >= draw_frame_count:
        return float(len(model.years))
    return min(len(model.years), frame_index / max(1, draw_frame_count - 1) * len(model.years))


def draw_completed_boundaries(
    draw: ImageDraw.ImageDraw,
    model: RingModel,
    config: RenderConfig,
    current_index: float,
) -> None:
    completed = min(len(model.years), int(math.floor(current_index)))
    for boundary_index in range(1, completed + 1):
        year = model.years[boundary_index - 1]
        if year % 5 == 0 or boundary_index == completed:
            alpha, width = 188, 2
        else:
            alpha, width = 132, 1
        draw.line(
            closed(boundary_points(model, config, boundary_index, samples=560)),
            fill=config.neutral_rgb + (alpha,),
            width=width,
        )


def draw_growth_front(
    draw: ImageDraw.ImageDraw,
    model: RingModel,
    config: RenderConfig,
    current_index: float,
) -> None:
    if current_index >= len(model.years):
        draw.line(
            closed(boundary_points(model, config, len(model.years), samples=720)),
            fill=config.neutral_rgb + (235,),
            width=2,
        )
        return

    year_index = int(math.floor(current_index))
    frac = current_index - year_index
    points = []
    for sample_index in range(720):
        angle = sample_index * math.tau / 720
        inner = float(sample_periodic(model.boundaries[year_index], angle))
        outer = float(sample_periodic(model.boundaries[year_index + 1], angle))
        points.append(polar(config, inner + (outer - inner) * frac, angle))
    draw.line(closed(points), fill=config.neutral_rgb + (226,), width=2)


def draw_pith(draw: ImageDraw.ImageDraw, config: RenderConfig, frame_index: int) -> None:
    radius = config.pith_radius * smoothstep(min(1.0, frame_index / 18.0))
    if radius <= 0:
        return
    box = (
        config.center_x - radius,
        config.center_y - radius,
        config.center_x + radius,
        config.center_y + radius,
    )
    draw.ellipse(box, fill=config.neutral_rgb + (255,), outline=(112, 105, 96, 145), width=1)


def output_paths(config: RenderConfig) -> tuple[Path, Path, Path, Path]:
    raw_mp4 = config.output_dir / f"{config.name}.mp4"
    x_mp4 = config.output_dir / f"{config.name}_h264_aac.mp4"
    final_frame = config.output_dir / f"{config.name}_final_frame.png"
    contact_sheet = config.output_dir / f"{config.name}_contact_sheet.png"
    return raw_mp4, x_mp4, final_frame, contact_sheet


def render_all_frames(model: RingModel, config: RenderConfig, chunks: list[list[RectChunk]], frame_counts: np.ndarray) -> Image.Image:
    raw_mp4, _x_mp4, final_frame_path, _contact_sheet_path = output_paths(config)
    draw_frame_count = len(chunks)
    total_frame_count = draw_frame_count + max(0, int(round(config.hold_seconds * config.fps)))

    canvas = np.full((config.height, config.width, 3), 255, dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(raw_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        config.fps,
        (config.width, config.height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer.")

    final_frame: Image.Image | None = None
    contact_frames: list[Image.Image] = []
    contact_indices = {
        0,
        draw_frame_count // 3,
        draw_frame_count * 2 // 3,
        draw_frame_count - 1,
        max(0, total_frame_count - 1),
    }
    generated_count = 0
    for frame_index in range(total_frame_count):
        if frame_index < draw_frame_count:
            for chunk in chunks[frame_index]:
                plot_rect_cells(canvas, chunk, config)
            generated_count += int(frame_counts[frame_index])

        frame = Image.fromarray(canvas).convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")
        draw_pith(draw, config, min(frame_index, draw_frame_count - 1))
        idx_float = current_year_index(model, frame_index, draw_frame_count)
        draw_completed_boundaries(draw, model, config, idx_float)
        draw_growth_front(draw, model, config, idx_float)

        rgb = frame.convert("RGB")
        writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        final_frame = rgb
        if frame_index in contact_indices:
            contact_frames.append(rgb.copy())
        if frame_index % max(1, total_frame_count // 12) == 0:
            print(f"rendered {frame_index:04d}/{total_frame_count} cells={generated_count:,}/{model.total_cells:,}")

    writer.release()
    if final_frame is None:
        raise RuntimeError("No frames rendered.")
    final_frame.save(final_frame_path)
    if config.save_contact_sheet:
        save_contact_sheet(contact_frames, config)
    return final_frame


def save_contact_sheet(frames: list[Image.Image], config: RenderConfig) -> None:
    if not frames:
        return
    _raw_mp4, _x_mp4, _final_frame, contact_sheet_path = output_paths(config)
    font = load_font(14)
    thumb_w = 288
    thumb_h = int(thumb_w * config.height / config.width)
    padding = 18
    label_h = 24
    sheet = Image.new(
        "RGB",
        (thumb_w * len(frames) + padding * (len(frames) + 1), thumb_h + padding * 2 + label_h),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    duration = config.draw_seconds + config.hold_seconds
    labels = [f"{seconds:g}s" for seconds in np.linspace(0, duration, len(frames))]
    for index, frame in enumerate(frames):
        x = padding + index * (thumb_w + padding)
        y = padding
        sheet.paste(frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + thumb_h + 6), labels[index], font=font, fill=(96, 96, 90))
    sheet.save(contact_sheet_path)


def encode_h264_aac(config: RenderConfig) -> None:
    raw_mp4, x_mp4, _final_frame, _contact_sheet = output_paths(config)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_mp4),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(x_mp4),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)


def inspect_video(path: Path) -> list[str]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    completed = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], text=True, capture_output=True, check=False)
    return [
        line.strip()
        for line in completed.stderr.splitlines()
        if "Duration:" in line or "Video:" in line or "Audio:" in line
    ]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a dendrochronology-style mortality rings animation from time-series data."
    )
    parser.add_argument("--input", type=Path, default=Path("examples/belgium_weekly_deaths_1992_2025.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--name", default=None, help="Output filename prefix. Defaults to the input filename stem.")

    parser.add_argument("--date-column", default="date")
    parser.add_argument("--end-date-column", default="end_date")
    parser.add_argument("--year-column", default="year")
    parser.add_argument("--start-day-column", default="start_day")
    parser.add_argument("--end-day-column", default="end_day")
    parser.add_argument("--count-column", default="deaths")
    parser.add_argument("--baseline-column", default="baseline")
    parser.add_argument("--excess-column", default="excess")
    parser.add_argument("--day-indexing", choices=["one-based", "zero-based"], default="one-based")
    parser.add_argument("--period-days", type=int, default=None, help="Used with date input when no end date exists.")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--baseline-window", type=int, default=5)

    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--draw-seconds", type=float, default=51.0)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--radius-ratio", type=float, default=0.88)
    parser.add_argument("--pith-radius", type=float, default=17.0)
    parser.add_argument("--seed", type=int, default=20240615)

    parser.add_argument("--cell-unit", type=float, default=1.0, help="How many counted units each cell represents.")
    parser.add_argument("--max-cells", type=int, default=0, help="Optional cap; increases cell-unit automatically.")
    parser.add_argument("--thickness-gain", type=float, default=0.82)
    parser.add_argument("--thickness-response", type=float, default=0.30)
    parser.add_argument("--min-local-width", type=float, default=0.42)
    parser.add_argument("--max-local-width", type=float, default=2.25)
    parser.add_argument("--boundary-smoothing-days", type=float, default=2.2)
    parser.add_argument("--color-smoothing-days", type=float, default=1.0)
    parser.add_argument("--angle-jitter-days", type=float, default=0.70)
    parser.add_argument("--radial-jitter", type=float, default=0.08)
    parser.add_argument("--color-jitter", type=float, default=3.0)

    parser.add_argument("--palette", default=",".join(DEFAULT_PALETTE))
    parser.add_argument("--color-min", type=float, default=-0.50)
    parser.add_argument("--color-max", type=float, default=1.20)
    parser.add_argument("--neutral-color", default=DEFAULT_NEUTRAL)

    parser.add_argument("--no-h264", action="store_true", help="Skip the H.264/AAC social-platform encode.")
    parser.add_argument("--no-contact-sheet", action="store_true")
    return parser


def make_config(args: argparse.Namespace) -> RenderConfig:
    center_x = args.center_x if args.center_x is not None else args.width / 2.0
    center_y = args.center_y if args.center_y is not None else args.height / 2.0
    max_radius = min(args.width, args.height) * args.radius_ratio / 2.0
    name = args.name or args.input.stem
    return RenderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        draw_seconds=args.draw_seconds,
        hold_seconds=args.hold_seconds,
        center_x=center_x,
        center_y=center_y,
        pith_radius=args.pith_radius,
        max_radius=max_radius,
        seed=args.seed,
        neutral_rgb=hex_to_rgb(args.neutral_color),
        cell_unit=args.cell_unit,
        max_cells=args.max_cells,
        thickness_gain=args.thickness_gain,
        thickness_response=args.thickness_response,
        min_local_width=args.min_local_width,
        max_local_width=args.max_local_width,
        boundary_smoothing_days=args.boundary_smoothing_days,
        color_smoothing_days=args.color_smoothing_days,
        angle_jitter_days=args.angle_jitter_days,
        radial_jitter=args.radial_jitter,
        color_jitter=args.color_jitter,
        output_dir=args.output_dir,
        name=name,
        encode_h264=not args.no_h264,
        save_contact_sheet=not args.no_contact_sheet,
    )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    config = make_config(args)
    color_stops, palette = parse_palette(args.palette, args.color_min, args.color_max)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, config)
    print(f"years: {model.years[0]}-{model.years[-1]} ({len(model.years)} rings)")
    print(f"input total: {model.total_count:,.0f}")
    print(f"cells: {model.total_cells:,} (1 cell = {model.cell_unit:,.4g} counted units)")
    print(f"format: {config.width}x{config.height}, {config.draw_seconds:g}s draw, {config.fps} fps")

    chunks, frame_counts = generate_frame_chunks(model, config, color_stops, palette)
    render_all_frames(model, config, chunks, frame_counts)
    if config.encode_h264:
        encode_h264_aac(config)

    raw_mp4, x_mp4, final_frame, contact_sheet = output_paths(config)
    print(f"video: {raw_mp4} ({raw_mp4.stat().st_size / 1024 / 1024:.2f} MB)")
    if config.encode_h264:
        print(f"h264/aac video: {x_mp4} ({x_mp4.stat().st_size / 1024 / 1024:.2f} MB)")
        for line in inspect_video(x_mp4):
            print(f"  {line}")
    print(f"final frame: {final_frame}")
    if config.save_contact_sheet:
        print(f"contact sheet: {contact_sheet}")


if __name__ == "__main__":
    main()
