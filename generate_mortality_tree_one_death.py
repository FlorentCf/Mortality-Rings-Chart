from __future__ import annotations

import calendar
import json
import math
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 24
DURATION_SECONDS = 30
FRAME_COUNT = FPS * DURATION_SECONDS

SEED = 20240612
ANGLE_SAMPLES = 1440

TREE_CENTER = (650, 540)
PITH_RADIUS = 14.0
MAX_RADIUS = 445.0
ANGLE_OFFSET = -math.pi / 2.0 + math.radians(23.0)

BACKGROUND = (255, 255, 255)
INK = (35, 35, 35)
MID = (105, 105, 105)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
QUERY_FILE = Path(r"C:\Users\Florent\Downloads\pxapi-api_table_px-x-0102020206_111.px.json")
PXWEB_URL = "https://www.pxweb.bfs.admin.ch/api/v1/en/px-x-0102020206_111/px-x-0102020206_111.px"

RAW_JSON_PATH = OUTPUT_DIR / "bfs_monthly_deaths_jsonstat.json"
DAILY_CSV_PATH = OUTPUT_DIR / "bfs_smoothed_daily_deaths_1877_2024.csv"
MP4_PATH = OUTPUT_DIR / "swiss_mortality_tree_1877_2024_one_death_smoothed_30s.mp4"
X_MP4_PATH = OUTPUT_DIR / "swiss_mortality_tree_1877_2024_one_death_smoothed_x_h264_aac.mp4"
FINAL_FRAME_PATH = OUTPUT_DIR / "swiss_mortality_tree_1877_2024_one_death_smoothed_final_frame.png"
CONTACT_SHEET_PATH = OUTPUT_DIR / "swiss_mortality_tree_1877_2024_one_death_smoothed_contact_sheet.png"

PALETTE_STOPS = [
    (-0.50, "#1B263B"),
    (-0.35, "#2F5266"),
    (-0.20, "#6F8F91"),
    (-0.10, "#B2B8AA"),
    (0.00, "#E7DDC8"),
    (0.10, "#D6A84F"),
    (0.25, "#B7653A"),
    (0.50, "#872B36"),
    (0.80, "#4A1022"),
    (1.20, "#1F0610"),
]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_names = [
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(38, bold=True)
FONT_SECTION = load_font(28, bold=True)
FONT_BODY = load_font(21)
FONT_SMALL = load_font(16)
FONT_TINY = load_font(13)


@dataclass(slots=True)
class DayRecord:
    year_index: int
    year: int
    month: int
    day: int
    day_of_year: int
    days_in_year: int
    deaths: int
    smooth_rate: float
    baseline_rate: float
    excess: float


@dataclass(slots=True)
class MortalityModel:
    years: list[int]
    days_by_year: list[list[DayRecord]]
    boundaries: list[np.ndarray]
    total_deaths: int
    source: str


@dataclass(slots=True)
class FrameChunk:
    x: np.ndarray
    y: np.ndarray
    rgb: np.ndarray


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    clean = value.strip().lstrip("#")
    return tuple(int(clean[index : index + 2], 16) for index in (0, 2, 4))


PALETTE_VALUES = np.array([stop for stop, _color in PALETTE_STOPS], dtype=float)
PALETTE_COLORS = np.array([hex_to_rgb(color) for _stop, color in PALETTE_STOPS], dtype=float)


def color_for_excess(excess: float) -> np.ndarray:
    clamped = float(np.clip(excess, PALETTE_VALUES[0], PALETTE_VALUES[-1]))
    r = np.interp(clamped, PALETTE_VALUES, PALETTE_COLORS[:, 0])
    g = np.interp(clamped, PALETTE_VALUES, PALETTE_COLORS[:, 1])
    b = np.interp(clamped, PALETTE_VALUES, PALETTE_COLORS[:, 2])
    return np.array([r, g, b], dtype=np.float32)


def colors_for_excess(excess: np.ndarray) -> np.ndarray:
    clamped = np.clip(excess, PALETTE_VALUES[0], PALETTE_VALUES[-1])
    r = np.interp(clamped, PALETTE_VALUES, PALETTE_COLORS[:, 0])
    g = np.interp(clamped, PALETTE_VALUES, PALETTE_COLORS[:, 1])
    b = np.interp(clamped, PALETTE_VALUES, PALETTE_COLORS[:, 2])
    return np.column_stack((r, g, b)).astype(np.float32)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def polar(radius: float, angle: float, center: tuple[int, int] = TREE_CENTER) -> tuple[float, float]:
    visual_angle = angle + ANGLE_OFFSET
    return (
        center[0] + radius * math.cos(visual_angle),
        center[1] + radius * math.sin(visual_angle),
    )


def closed(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points + [points[0]] if points else points


def circular_gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(2, int(math.ceil(sigma * 4.0)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    result = np.zeros_like(values, dtype=float)
    for weight, offset in zip(kernel, offsets):
        result += weight * np.roll(values, int(offset))
    return result


def periodic_interp(xp: np.ndarray, fp: np.ndarray, x: np.ndarray, period: float) -> np.ndarray:
    xp_ext = np.r_[xp - period, xp, xp + period]
    fp_ext = np.r_[fp, fp, fp]
    return np.interp(x, xp_ext, fp_ext)


def distribute_integer(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(weights.astype(float), 1e-9)
    shares = weights / weights.sum() * total
    values = np.floor(shares).astype(int)
    remainder = int(total - values.sum())
    if remainder > 0:
        order = np.argsort(-(shares - values))
        values[order[:remainder]] += 1
    return values


def fetch_pxweb_data() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    query_obj = json.loads(QUERY_FILE.read_text(encoding="utf-8-sig"))["queryObj"]
    body = json.dumps(query_obj).encode("utf-8")
    request = urllib.request.Request(
        PXWEB_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    RAW_JSON_PATH.write_bytes(raw)
    return json.loads(raw.decode("utf-8"))


def parse_monthly_records(data: dict) -> tuple[list[int], np.ndarray, str]:
    dataset = data["dataset"]
    year_dim = dataset["dimension"]["Jahr"]["category"]["index"]
    indicator_dim = dataset["dimension"]["Demografisches Merkmal und Indikator"]["category"]["index"]
    years = [int(year) for year in year_dim.keys()]
    month_codes = list(indicator_dim.keys())
    values = np.array([np.nan if value is None else float(value) for value in dataset["value"]])
    monthly = values.reshape((len(years), len(month_codes))).astype(float)

    complete_mask = ~np.isnan(monthly).any(axis=1)
    complete_years = [year for year, complete in zip(years, complete_mask) if complete]
    complete_monthly = monthly[complete_mask].astype(int)
    return complete_years, complete_monthly, str(dataset.get("source", "FSO"))


def compute_monthly_baseline(monthly: np.ndarray) -> np.ndarray:
    baseline = np.zeros_like(monthly, dtype=float)
    for year_index in range(monthly.shape[0]):
        for month_index in range(12):
            start = max(0, year_index - 10)
            previous = monthly[start:year_index, month_index]
            if len(previous) >= 5:
                baseline[year_index, month_index] = float(np.mean(previous))
            else:
                end = min(monthly.shape[0], 10)
                baseline[year_index, month_index] = float(np.mean(monthly[:end, month_index]))
    return np.maximum(baseline, 1.0)


def month_midpoints(year: int) -> tuple[np.ndarray, np.ndarray, int]:
    days_in_month = np.array([calendar.monthrange(year, month)[1] for month in range(1, 13)], dtype=float)
    starts = np.r_[0.0, np.cumsum(days_in_month)[:-1]]
    return starts + days_in_month / 2.0, days_in_month, int(days_in_month.sum())


def daily_curve_from_monthly(
    year: int,
    monthly_values: np.ndarray,
    rng: np.random.Generator,
    add_texture: bool,
) -> tuple[np.ndarray, np.ndarray]:
    midpoints, days_in_month, days_in_year = month_midpoints(year)
    rates = monthly_values.astype(float) / days_in_month
    x = np.arange(days_in_year, dtype=float) + 0.5

    curve = periodic_interp(midpoints, rates, x, float(days_in_year))
    curve = circular_gaussian(curve, sigma=10.0)

    if add_texture:
        short_noise = rng.normal(0.0, 1.0, days_in_year)
        short_noise = circular_gaussian(short_noise, sigma=rng.uniform(2.0, 4.0))
        short_noise = (short_noise - short_noise.mean()) / max(1e-9, short_noise.std())

        long_noise = rng.normal(0.0, 1.0, days_in_year)
        long_noise = circular_gaussian(long_noise, sigma=rng.uniform(16.0, 28.0))
        long_noise = (long_noise - long_noise.mean()) / max(1e-9, long_noise.std())

        pulse = np.ones(days_in_year)
        for _ in range(rng.integers(2, 5)):
            center = rng.uniform(0.0, days_in_year)
            width = rng.uniform(4.0, 16.0)
            distance = np.minimum(np.abs(x - center), days_in_year - np.abs(x - center))
            pulse += rng.uniform(-0.030, 0.055) * np.exp(-0.5 * (distance / width) ** 2)

        curve *= np.clip(1.0 + 0.035 * short_noise + 0.030 * long_noise, 0.82, 1.22)
        curve *= np.clip(pulse, 0.88, 1.15)
        curve = circular_gaussian(curve, sigma=3.5)

    curve = np.maximum(curve, 0.001)
    curve *= float(monthly_values.sum()) / curve.sum()
    daily_deaths = distribute_integer(int(monthly_values.sum()), curve)
    return daily_deaths, curve


def build_boundaries(days_by_year: list[list[DayRecord]]) -> list[np.ndarray]:
    boundaries = [np.full(ANGLE_SAMPLES, PITH_RADIUS, dtype=float)]
    annual_totals = np.array([sum(day.deaths for day in days) for days in days_by_year], dtype=float)
    year_widths = (annual_totals / annual_totals.mean()) ** 0.45
    year_widths = year_widths / year_widths.sum() * (MAX_RADIUS - PITH_RADIUS)

    for year_index, days in enumerate(days_by_year):
        daily_rates = np.array([day.smooth_rate for day in days], dtype=float)
        daily_rates = circular_gaussian(daily_rates, sigma=5.0)
        local = np.zeros(ANGLE_SAMPLES, dtype=float)
        for sample_index in range(ANGLE_SAMPLES):
            day_position = sample_index / ANGLE_SAMPLES * len(days)
            left = int(math.floor(day_position)) % len(days)
            right = (left + 1) % len(days)
            frac = day_position - math.floor(day_position)
            local[sample_index] = daily_rates[left] * (1.0 - frac) + daily_rates[right] * frac
        local = (local / max(1e-9, local.mean())) ** 0.50
        local = circular_gaussian(local, sigma=9.0)
        local = np.clip(local, 0.68, 1.55)
        local /= local.mean()
        boundaries.append(boundaries[-1] + year_widths[year_index] * local)

    return boundaries


def write_daily_csv(days_by_year: list[list[DayRecord]]) -> None:
    lines = ["year,month,day,day_of_year,deaths,smoothed_rate,baseline_rate,excess\n"]
    for days in days_by_year:
        for day in days:
            lines.append(
                f"{day.year},{day.month},{day.day},{day.day_of_year + 1},{day.deaths},"
                f"{day.smooth_rate:.6f},{day.baseline_rate:.6f},{day.excess:.6f}\n"
            )
    DAILY_CSV_PATH.write_text("".join(lines), encoding="utf-8")


def build_model() -> MortalityModel:
    data = fetch_pxweb_data()
    years, monthly, source = parse_monthly_records(data)
    baseline_monthly = compute_monthly_baseline(monthly)
    days_by_year: list[list[DayRecord]] = []

    for year_index, year in enumerate(years):
        rng = np.random.default_rng(SEED + year)
        daily_deaths, smooth_rate = daily_curve_from_monthly(year, monthly[year_index], rng, add_texture=True)
        _baseline_counts, baseline_rate = daily_curve_from_monthly(
            year,
            baseline_monthly[year_index],
            np.random.default_rng(SEED + year + 500_000),
            add_texture=False,
        )

        year_days: list[DayRecord] = []
        day_cursor = 0
        for month in range(1, 13):
            month_days = calendar.monthrange(year, month)[1]
            for day in range(1, month_days + 1):
                excess = smooth_rate[day_cursor] / max(1e-9, baseline_rate[day_cursor]) - 1.0
                year_days.append(
                    DayRecord(
                        year_index=year_index,
                        year=year,
                        month=month,
                        day=day,
                        day_of_year=day_cursor,
                        days_in_year=len(daily_deaths),
                        deaths=int(daily_deaths[day_cursor]),
                        smooth_rate=float(smooth_rate[day_cursor]),
                        baseline_rate=float(baseline_rate[day_cursor]),
                        excess=float(excess),
                    )
                )
                day_cursor += 1
        days_by_year.append(year_days)

    boundaries = build_boundaries(days_by_year)
    write_daily_csv(days_by_year)
    total_deaths = int(sum(day.deaths for days in days_by_year for day in days))
    return MortalityModel(years=years, days_by_year=days_by_year, boundaries=boundaries, total_deaths=total_deaths, source=source)


MODEL = build_model()


def sample_periodic(values: np.ndarray, angle: np.ndarray | float) -> np.ndarray | float:
    position = (np.asarray(angle) % math.tau) / math.tau * len(values)
    left = np.floor(position).astype(int) % len(values)
    right = (left + 1) % len(values)
    frac = position - np.floor(position)
    sampled = values[left] * (1.0 - frac) + values[right] * frac
    if np.isscalar(angle):
        return float(sampled)
    return sampled


def radius_at(year_index: int, radial_t: np.ndarray, angle: np.ndarray) -> np.ndarray:
    inner = sample_periodic(MODEL.boundaries[year_index], angle)
    outer = sample_periodic(MODEL.boundaries[year_index + 1], angle)
    return inner + (outer - inner) * radial_t


def boundary_points(year_boundary_index: int, samples: int = 720) -> list[tuple[float, float]]:
    values = MODEL.boundaries[year_boundary_index]
    phase = (year_boundary_index * 2.399963229728653) % math.tau
    return [
        polar(float(sample_periodic(values, phase + i * math.tau / samples)), phase + i * math.tau / samples)
        for i in range(samples)
    ]


def plot_cells(canvas: np.ndarray, chunk: FrameChunk) -> None:
    x = chunk.x
    y = chunk.y
    valid = (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)
    if not np.any(valid):
        return
    xv = x[valid]
    yv = y[valid]
    rgb = chunk.rgb[valid]
    canvas[yv, xv] = rgb


def generate_frame_chunks() -> tuple[list[list[FrameChunk]], np.ndarray]:
    chunks: list[list[FrameChunk]] = [[] for _ in range(FRAME_COUNT)]
    frame_counts = np.zeros(FRAME_COUNT, dtype=np.int64)
    year_count = len(MODEL.years)

    for year_index, days in enumerate(MODEL.days_by_year):
        excess_values = np.array([day.excess for day in days], dtype=float)
        day_positions = np.arange(len(days), dtype=float) + 0.5
        for day in days:
            n = day.deaths
            if n <= 0:
                continue

            rng = np.random.default_rng(SEED + day.year * 10_000 + day.day_of_year)
            day_fraction = (day.day_of_year + 0.5) / day.days_in_year
            theta_center = math.tau * day_fraction
            sigma_days = rng.uniform(6.0, 12.0)
            theta = theta_center + rng.normal(0.0, math.tau * sigma_days / day.days_in_year, n)

            radial_t = rng.random(n)
            theta += (radial_t - 0.5) * rng.normal(0.0, math.tau * 2.4 / day.days_in_year)
            theta += rng.normal(0.0, 0.00075, n)
            radius = radius_at(year_index, radial_t, theta) + rng.normal(0.0, 0.10, n)
            visual_angle = theta + ANGLE_OFFSET
            x = np.rint(TREE_CENTER[0] + radius * np.cos(visual_angle) + rng.normal(0.0, 0.42, n)).astype(np.int32)
            y = np.rint(TREE_CENTER[1] + radius * np.sin(visual_angle) + rng.normal(0.0, 0.42, n)).astype(np.int32)

            color_position = (theta % math.tau) / math.tau * day.days_in_year
            cell_excess = periodic_interp(day_positions, excess_values, color_position, float(day.days_in_year))
            base_color = colors_for_excess(cell_excess)
            jitter = rng.normal(0.0, 5.5, (n, 3))
            rgb = np.clip(base_color + jitter, 0, 255).astype(np.uint8)

            progress = (
                year_index + (day.day_of_year + rng.random(n)) / day.days_in_year
            ) / year_count
            birth_frames = np.clip((progress * (FRAME_COUNT - 1)).astype(np.int32), 0, FRAME_COUNT - 1)

            for frame_index in np.unique(birth_frames):
                mask = birth_frames == frame_index
                chunk = FrameChunk(x=x[mask], y=y[mask], rgb=rgb[mask])
                chunks[int(frame_index)].append(chunk)
                frame_counts[int(frame_index)] += int(mask.sum())

        if year_index % 20 == 0:
            print(f"prepared cells through {MODEL.years[year_index]}")

    return chunks, frame_counts


def current_year_index(frame_index: int) -> float:
    return min(len(MODEL.years), frame_index / max(1, FRAME_COUNT - 1) * len(MODEL.years))


def draw_completed_boundaries(draw: ImageDraw.ImageDraw, current_index: float) -> None:
    completed = min(len(MODEL.years), int(math.floor(current_index)))
    for boundary_index in range(1, completed + 1):
        year = MODEL.years[boundary_index - 1]
        if year in (1918, 2020):
            shade, alpha, width = 38, 175, 2
        elif year % 20 == 0 or boundary_index == completed:
            shade, alpha, width = 84, 105, 1
        else:
            shade, alpha, width = 130, 30, 1
        draw.line(closed(boundary_points(boundary_index, samples=540)), fill=(shade, shade, shade, alpha), width=width)


def draw_growth_front(draw: ImageDraw.ImageDraw, current_index: float) -> None:
    if current_index >= len(MODEL.years):
        draw.line(closed(boundary_points(len(MODEL.years), samples=720)), fill=(42, 42, 42, 210), width=2)
        return

    year_index = int(math.floor(current_index))
    frac = current_index - year_index
    points = []
    for i in range(720):
        angle = i * math.tau / 720
        inner = float(sample_periodic(MODEL.boundaries[year_index], angle))
        outer = float(sample_periodic(MODEL.boundaries[year_index + 1], angle))
        points.append(polar(inner + (outer - inner) * frac, angle))
    draw.line(closed(points), fill=(42, 42, 42, 185), width=2)


def draw_month_ticks(draw: ImageDraw.ImageDraw) -> None:
    return


def draw_color_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.text((x, y), "excess mortality palette", font=FONT_SMALL, fill=MID)
    legend_w = 330
    legend_h = 20
    for i in range(legend_w):
        excess = np.interp(i, [0, legend_w - 1], [PALETTE_VALUES[0], PALETTE_VALUES[-1]])
        color = tuple(int(v) for v in color_for_excess(float(excess)))
        draw.line((x + i, y + 32, x + i, y + 32 + legend_h), fill=color)
    draw.rectangle((x, y + 32, x + legend_w, y + 32 + legend_h), outline=(145, 145, 145), width=1)
    draw.text((x, y + 60), "-50%", font=FONT_TINY, fill=MID)
    draw.text((x + 143, y + 60), "0%", font=FONT_TINY, fill=MID)
    draw.text((x + 292, y + 60), "+120%", font=FONT_TINY, fill=MID)


def draw_year_rail(draw: ImageDraw.ImageDraw, x: int, top: int, bottom: int, idx_float: float) -> None:
    draw.line((x, top, x, bottom), fill=(174, 174, 174, 170), width=2)
    start = MODEL.years[0]
    end = MODEL.years[-1]
    for year in range(1880, 2030, 20):
        if year < start or year > end or any(abs(year - event) <= 2 for event in (1918, 2020)):
            continue
        fraction = (year - start) / (end - start)
        y = top + fraction * (bottom - top)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(85, 85, 85, 210))
        draw.text((x + 18, y - 9), str(year), font=FONT_TINY, fill=MID)

    for event_year in (1918, 2020):
        fraction = (event_year - start) / (end - start)
        y = top + fraction * (bottom - top)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(35, 35, 35, 235))
        label_y = y - 28 if event_year == 2020 else y - 10
        draw.text((x + 18, label_y), str(event_year), font=FONT_SMALL, fill=INK)

    marker_y = top + min(1.0, max(0.0, idx_float / len(MODEL.years))) * (bottom - top)
    draw.rounded_rectangle((x - 20, marker_y - 10, x + 20, marker_y + 10), radius=10, fill=(54, 54, 54, 235))
    marker_label_y = marker_y + 18 if idx_float > len(MODEL.years) - 5 else marker_y - 10
    draw.text((x + 56, marker_label_y), "growth front", font=FONT_SMALL, fill=INK)


def draw_annotations(
    draw: ImageDraw.ImageDraw,
    frame_index: int,
    generated_count: int,
    new_cells: int,
) -> None:
    idx_float = current_year_index(frame_index)
    year_index = min(len(MODEL.years) - 1, int(math.floor(idx_float)))
    year = MODEL.years[year_index]
    annual_deaths = sum(day.deaths for day in MODEL.days_by_year[year_index])
    year_excess = np.average(
        [day.excess for day in MODEL.days_by_year[year_index]],
        weights=[max(1, day.deaths) for day in MODEL.days_by_year[year_index]],
    )

    panel_x = 1240
    draw.text((panel_x, 98), "Swiss mortality rings", font=FONT_TITLE, fill=INK)
    draw.text((panel_x, 143), "one death per cell, smoothed into daily growth", font=FONT_BODY, fill=MID)

    draw.text((panel_x, 210), f"{year}", font=FONT_SECTION, fill=INK)
    draw.text((panel_x + 92, 216), f"{annual_deaths:,} deaths", font=FONT_BODY, fill=INK)
    draw.text((panel_x, 248), f"{year_excess:+.1%} vs baseline", font=FONT_BODY, fill=MID)

    stat_y = 318
    draw.text((panel_x, stat_y), "death cells generated", font=FONT_SMALL, fill=MID)
    draw.text((panel_x + 210, stat_y - 7), f"{generated_count:,} / {MODEL.total_deaths:,}", font=FONT_SECTION, fill=INK)
    draw.text((panel_x, stat_y + 44), "scale", font=FONT_SMALL, fill=MID)
    draw.text((panel_x + 210, stat_y + 37), "1 cell = 1 death", font=FONT_BODY, fill=INK)
    draw.text((panel_x, stat_y + 80), "new this frame", font=FONT_SMALL, fill=MID)
    draw.text((panel_x + 210, stat_y + 73), f"{new_cells:,}", font=FONT_BODY, fill=INK)

    draw_color_legend(draw, panel_x, 478)
    draw_year_rail(draw, panel_x + 36, 650, 920, idx_float)

    draw.line((82, 1000, 1098, 1000), fill=(214, 214, 214, 200), width=1)
    note = "Angle = time within year; monthly deaths are smoothed into a continuous daily field; each plotted cell is one death."
    draw.text((82, 1024), note, font=FONT_SMALL, fill=(118, 118, 118))
    source = "Source: Swiss Federal Statistical Office PXWEB, deaths per month and mortality since 1803."
    draw.text((1240, 1000), source, font=FONT_TINY, fill=(130, 130, 130))


def draw_pith(draw: ImageDraw.ImageDraw, frame_index: int) -> None:
    progress = smoothstep(min(1.0, frame_index / 30.0))
    radius = PITH_RADIUS * progress
    if radius <= 0:
        return
    bbox = (
        TREE_CENTER[0] - radius,
        TREE_CENTER[1] - radius,
        TREE_CENTER[0] + radius,
        TREE_CENTER[1] + radius,
    )
    draw.ellipse(bbox, fill=(231, 221, 200, 255), outline=(112, 105, 96, 145), width=1)


def render_all_frames(chunks: list[list[FrameChunk]], frame_counts: np.ndarray) -> tuple[Image.Image, list[Image.Image]]:
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    final_frame: Image.Image | None = None
    contact_frames: list[Image.Image] = []
    contact_indices = {0, FRAME_COUNT // 4, FRAME_COUNT // 2, FRAME_COUNT * 3 // 4, FRAME_COUNT - 1}
    generated_count = 0

    writer = cv2.VideoWriter(str(MP4_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer.")

    for frame_index in range(FRAME_COUNT):
        for chunk in chunks[frame_index]:
            plot_cells(canvas, chunk)
        generated_count += int(frame_counts[frame_index])

        frame = Image.fromarray(canvas, mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")
        draw_pith(draw, frame_index)
        idx_float = current_year_index(frame_index)
        draw_completed_boundaries(draw, idx_float)
        draw_growth_front(draw, idx_float)
        draw_month_ticks(draw)
        draw_annotations(draw, frame_index, generated_count=generated_count, new_cells=int(frame_counts[frame_index]))

        rgb = frame.convert("RGB")
        writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        final_frame = rgb
        if frame_index in contact_indices:
            contact_frames.append(rgb.copy())

        if frame_index % max(1, FRAME_COUNT // 10) == 0:
            print(f"rendered {frame_index:04d}/{FRAME_COUNT} cells={generated_count:,}/{MODEL.total_deaths:,}")

    writer.release()
    if final_frame is None:
        raise RuntimeError("No frames rendered.")
    return final_frame, contact_frames


def save_contact_sheet(frames: list[Image.Image]) -> None:
    if not frames:
        return
    thumb_w = 384
    thumb_h = 216
    padding = 18
    label_h = 24
    sheet = Image.new("RGB", (thumb_w * len(frames) + padding * (len(frames) + 1), thumb_h + padding * 2 + label_h), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    labels = ["0s", "7.5s", "15s", "22.5s", "30s"]
    for index, frame in enumerate(frames):
        x = padding + index * (thumb_w + padding)
        y = padding
        sheet.paste(frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + thumb_h + 6), labels[index], font=FONT_SMALL, fill=MID)
    sheet.save(CONTACT_SHEET_PATH)


def encode_x_compatible() -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(MP4_PATH),
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
        "27",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(X_MP4_PATH),
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"years: {MODEL.years[0]}-{MODEL.years[-1]} ({len(MODEL.years)} rings)")
    print(f"total deaths/cells: {MODEL.total_deaths:,}")
    print("scale: 1 cell = 1 death")
    print("smoothing: monthly rates -> periodic daily interpolation -> gaussian smoothing -> annual total preserved")
    chunks, frame_counts = generate_frame_chunks()
    final_frame, contact_frames = render_all_frames(chunks, frame_counts)
    final_frame.save(FINAL_FRAME_PATH)
    save_contact_sheet(contact_frames)
    encode_x_compatible()
    print(f"video: {MP4_PATH} ({MP4_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"x video: {X_MP4_PATH} ({X_MP4_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"final frame: {FINAL_FRAME_PATH}")
    print(f"contact sheet: {CONTACT_SHEET_PATH}")
    for line in inspect_video(X_MP4_PATH):
        print(f"  {line}")


if __name__ == "__main__":
    main()
