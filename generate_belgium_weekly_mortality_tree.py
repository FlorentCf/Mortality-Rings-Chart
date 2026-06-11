from __future__ import annotations

import calendar
import math
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 24
SECONDS_PER_YEAR = 3
DURATION_SECONDS = 102
FRAME_COUNT = FPS * DURATION_SECONDS

SEED = 20240613
ANGLE_SAMPLES = 1440
WEEK_COUNT = 52

TREE_CENTER = (650, 540)
PITH_RADIUS = 18.0
MAX_RADIUS = 420.0
ANGLE_OFFSET = -math.pi / 2.0 + math.radians(17.0)

BACKGROUND = (255, 255, 255)
INK = (35, 35, 35)
MID = (105, 105, 105)
NEUTRAL_RING = (231, 221, 200)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
STATBEL_DIR = OUTPUT_DIR / "statbel"
STATBEL_ZIP = STATBEL_DIR / "TF_DEATHS.zip"
STATBEL_URL = "https://statbel.fgov.be/sites/default/files/files/opendata/bevolking/TF_DEATHS.zip"

WEEKLY_CSV_PATH = OUTPUT_DIR / "belgium_weekly_deaths_1992_2025.csv"
MP4_PATH = OUTPUT_DIR / "belgium_weekly_mortality_tree_1992_2025_radial_growth_3s_per_year.mp4"
X_MP4_PATH = OUTPUT_DIR / "belgium_weekly_mortality_tree_1992_2025_radial_growth_3s_per_year_x_h264_aac.mp4"
FINAL_FRAME_PATH = OUTPUT_DIR / "belgium_weekly_mortality_tree_1992_2025_radial_growth_3s_per_year_final_frame.png"
CONTACT_SHEET_PATH = OUTPUT_DIR / "belgium_weekly_mortality_tree_1992_2025_radial_growth_3s_per_year_contact_sheet.png"

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
class WeekRecord:
    year_index: int
    year: int
    week_index: int
    start_day: int
    end_day: int
    deaths: int
    baseline: float
    excess: float
    days: int


@dataclass(slots=True)
class DayRecord:
    year_index: int
    year: int
    day_of_year: int
    days_in_year: int
    deaths: int
    week_index: int
    weekly_excess: float


@dataclass(slots=True)
class BelgiumModel:
    years: list[int]
    weeks_by_year: list[list[WeekRecord]]
    days_by_year: list[list[DayRecord]]
    boundaries: list[np.ndarray]
    total_deaths: int


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


def circular_gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
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


def closed(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points + [points[0]] if points else points


def download_statbel_zip() -> None:
    STATBEL_DIR.mkdir(parents=True, exist_ok=True)
    if not STATBEL_ZIP.exists():
        urllib.request.urlretrieve(STATBEL_URL, STATBEL_ZIP)


def load_daily_data() -> pd.DataFrame:
    download_statbel_zip()
    with ZipFile(STATBEL_ZIP) as archive:
        with archive.open("TF_DEATHS.txt") as file:
            df = pd.read_csv(file, sep="|")

    df["date"] = pd.to_datetime(df["DATE_DEATH"], dayfirst=True)
    df["year"] = df["date"].dt.year.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int) - 1
    df = df[(df["year"] >= 1992) & (df["year"] <= 2025)].copy()
    return df.sort_values("date").reset_index(drop=True)


def week_index_for_day(day_of_year: int, days_in_year: int) -> int:
    return min(WEEK_COUNT - 1, int(day_of_year / days_in_year * WEEK_COUNT))


def compute_weekly_baseline(weekly_matrix: np.ndarray) -> np.ndarray:
    baseline = np.zeros_like(weekly_matrix, dtype=float)
    for year_index in range(weekly_matrix.shape[0]):
        for week_index in range(weekly_matrix.shape[1]):
            start = max(0, year_index - 5)
            previous = weekly_matrix[start:year_index, week_index]
            previous = previous[previous > 0]
            if len(previous) >= 3:
                value = float(np.mean(previous))
            else:
                end = min(weekly_matrix.shape[0], max(8, year_index + 6))
                peers = weekly_matrix[:end, week_index]
                peers = peers[peers > 0]
                value = float(np.mean(peers)) if len(peers) else 1.0
            baseline[year_index, week_index] = max(1.0, value)

    for year_index in range(baseline.shape[0]):
        baseline[year_index] = circular_gaussian(baseline[year_index], sigma=0.80)
    return baseline


def build_model() -> BelgiumModel:
    df = load_daily_data()
    years = sorted(df["year"].unique().tolist())
    year_to_index = {year: index for index, year in enumerate(years)}
    weekly_matrix = np.zeros((len(years), WEEK_COUNT), dtype=float)
    week_days_matrix = np.zeros((len(years), WEEK_COUNT), dtype=float)

    for row in df.itertuples(index=False):
        days_in_year = 366 if calendar.isleap(int(row.year)) else 365
        week_index = week_index_for_day(int(row.day_of_year), days_in_year)
        weekly_matrix[year_to_index[int(row.year)], week_index] += int(row.CNT)
        week_days_matrix[year_to_index[int(row.year)], week_index] += 1

    weekly_rate_matrix = np.divide(
        weekly_matrix * 7.0,
        np.maximum(1.0, week_days_matrix),
        out=np.zeros_like(weekly_matrix),
        where=week_days_matrix > 0,
    )
    baseline_rate = compute_weekly_baseline(weekly_rate_matrix)
    weeks_by_year: list[list[WeekRecord]] = []
    days_by_year: list[list[DayRecord]] = []

    for year_index, year in enumerate(years):
        days_in_year = 366 if calendar.isleap(year) else 365
        year_weeks: list[WeekRecord] = []
        for week_index in range(WEEK_COUNT):
            deaths = int(round(weekly_matrix[year_index, week_index]))
            if deaths <= 0:
                continue
            days_in_bin = int(week_days_matrix[year_index, week_index])
            rate = deaths / max(1, days_in_bin) * 7.0
            expected_rate = float(baseline_rate[year_index, week_index])
            excess = rate / expected_rate - 1.0
            start_day = int(math.floor(week_index * days_in_year / WEEK_COUNT))
            end_day = int(math.floor((week_index + 1) * days_in_year / WEEK_COUNT) - 1)
            year_weeks.append(
                WeekRecord(year_index, year, week_index, start_day, end_day, deaths, expected_rate, excess, days_in_bin)
            )
        weeks_by_year.append(year_weeks)

        year_df = df[df["year"] == year]
        year_days: list[DayRecord] = []
        for row in year_df.itertuples(index=False):
            week_index = week_index_for_day(int(row.day_of_year), days_in_year)
            expected_rate = float(baseline_rate[year_index, week_index])
            days_in_bin = max(1.0, week_days_matrix[year_index, week_index])
            week_rate = weekly_matrix[year_index, week_index] / days_in_bin * 7.0
            weekly_excess = week_rate / expected_rate - 1.0
            year_days.append(
                DayRecord(
                    year_index=year_index,
                    year=year,
                    day_of_year=int(row.day_of_year),
                    days_in_year=days_in_year,
                    deaths=int(row.CNT),
                    week_index=week_index,
                    weekly_excess=float(weekly_excess),
                )
            )
        days_by_year.append(year_days)

    boundaries = build_boundaries(years, weeks_by_year)
    write_weekly_csv(weeks_by_year)
    total_deaths = int(df["CNT"].sum())
    return BelgiumModel(years, weeks_by_year, days_by_year, boundaries, total_deaths)


def build_boundaries(years: list[int], weeks_by_year: list[list[WeekRecord]]) -> list[np.ndarray]:
    annual_totals = np.array([sum(week.deaths for week in weeks) for weeks in weeks_by_year], dtype=float)
    mean_widths = (annual_totals / annual_totals.mean()) ** 0.35
    mean_widths = mean_widths / mean_widths.sum() * (MAX_RADIUS - PITH_RADIUS)
    boundaries = [np.full(ANGLE_SAMPLES, PITH_RADIUS, dtype=float)]

    for year_index, year in enumerate(years):
        days_in_year = 366 if calendar.isleap(year) else 365
        daily_factor = np.ones(days_in_year, dtype=float)
        weeks = weeks_by_year[year_index]
        week_rates = np.array([week.deaths / max(1, week.days) * 7.0 for week in weeks], dtype=float)
        week_mean = float(np.mean(week_rates))
        for week in weeks:
            week_rate = week.deaths / max(1, week.days) * 7.0
            count_factor = 0.46 * np.tanh((week_rate / max(1.0, week_mean) - 1.0) / 0.24)
            excess_factor = 0.56 * np.tanh(week.excess / 0.45)
            factor = 1.0 + count_factor + excess_factor
            daily_factor[week.start_day : week.end_day + 1] = factor

        # Keep real weekly bumps, but round their shoulders so they read as blobs.
        daily_factor = circular_gaussian(daily_factor, sigma=2.0)
        daily_factor = np.clip(daily_factor, 0.50, 2.15)
        daily_factor /= daily_factor.mean()

        local = np.zeros(ANGLE_SAMPLES, dtype=float)
        for sample_index in range(ANGLE_SAMPLES):
            day_position = sample_index / ANGLE_SAMPLES * days_in_year
            left = int(math.floor(day_position)) % days_in_year
            right = (left + 1) % days_in_year
            frac = day_position - math.floor(day_position)
            local[sample_index] = daily_factor[left] * (1.0 - frac) + daily_factor[right] * frac
        local = circular_gaussian(local, sigma=2.6)
        local /= local.mean()
        boundaries.append(boundaries[-1] + mean_widths[year_index] * local)

    return boundaries


def write_weekly_csv(weeks_by_year: list[list[WeekRecord]]) -> None:
    lines = ["year,week_index,start_day,end_day,deaths,baseline,excess\n"]
    for weeks in weeks_by_year:
        for week in weeks:
            lines.append(
                f"{week.year},{week.week_index},{week.start_day + 1},{week.end_day + 1},"
                f"{week.deaths},{week.baseline:.3f},{week.excess:.6f}\n"
            )
    WEEKLY_CSV_PATH.write_text("".join(lines), encoding="utf-8")


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


def boundary_points(boundary_index: int, samples: int = 720) -> list[tuple[float, float]]:
    values = MODEL.boundaries[boundary_index]
    phase = (boundary_index * 2.399963229728653) % math.tau
    return [
        polar(float(sample_periodic(values, phase + i * math.tau / samples)), phase + i * math.tau / samples)
        for i in range(samples)
    ]


def plot_cells(canvas: np.ndarray, chunk: FrameChunk) -> None:
    valid = (chunk.x >= 0) & (chunk.x < WIDTH) & (chunk.y >= 0) & (chunk.y < HEIGHT)
    if np.any(valid):
        canvas[chunk.y[valid], chunk.x[valid]] = chunk.rgb[valid]


def generate_frame_chunks() -> tuple[list[list[FrameChunk]], np.ndarray]:
    chunks: list[list[FrameChunk]] = [[] for _ in range(FRAME_COUNT)]
    frame_counts = np.zeros(FRAME_COUNT, dtype=np.int64)
    year_count = len(MODEL.years)

    for year_index, days in enumerate(MODEL.days_by_year):
        weekly_excess = np.zeros(WEEK_COUNT, dtype=float)
        for week in MODEL.weeks_by_year[year_index]:
            weekly_excess[week.week_index] = week.excess
        weekly_excess = circular_gaussian(weekly_excess, sigma=0.72)

        for day in days:
            n = day.deaths
            rng = np.random.default_rng(SEED + day.year * 10_000 + day.day_of_year)
            day_fraction = (day.day_of_year + 0.5) / day.days_in_year
            theta_center = math.tau * day_fraction

            # Weekly data should keep bumps/blobs, so this is only light placement jitter.
            theta = theta_center + rng.normal(0.0, math.tau * rng.uniform(0.38, 0.82) / day.days_in_year, n)
            radial_t = rng.random(n)
            theta += (radial_t - 0.5) * rng.normal(0.0, math.tau * 0.65 / day.days_in_year)

            radius = radius_at(year_index, radial_t, theta) + rng.normal(0.0, 0.11, n)
            visual_angle = theta + ANGLE_OFFSET
            x = np.rint(TREE_CENTER[0] + radius * np.cos(visual_angle) + rng.normal(0.0, 0.34, n)).astype(np.int32)
            y = np.rint(TREE_CENTER[1] + radius * np.sin(visual_angle) + rng.normal(0.0, 0.34, n)).astype(np.int32)

            week_float = (theta % math.tau) / math.tau * WEEK_COUNT
            excess = circular_interp(weekly_excess, week_float)
            base = colors_for_excess(excess)
            jitter = rng.normal(0.0, 5.0, (n, 3))
            rgb = np.clip(base + jitter, 0, 255).astype(np.uint8)

            radial_progress = np.clip(radial_t + rng.normal(0.0, 0.018, n), 0.0, 0.998)
            progress = (year_index + radial_progress) / year_count
            birth_frames = np.clip((progress * (FRAME_COUNT - 1)).astype(np.int32), 0, FRAME_COUNT - 1)
            for frame_index in np.unique(birth_frames):
                mask = birth_frames == frame_index
                chunks[int(frame_index)].append(FrameChunk(x=x[mask], y=y[mask], rgb=rgb[mask]))
                frame_counts[int(frame_index)] += int(mask.sum())

        if year_index % 5 == 0:
            print(f"prepared cells through {MODEL.years[year_index]}")

    return chunks, frame_counts


def current_year_index(frame_index: int) -> float:
    return min(len(MODEL.years), frame_index / max(1, FRAME_COUNT - 1) * len(MODEL.years))


def draw_completed_boundaries(draw: ImageDraw.ImageDraw, current_index: float) -> None:
    completed = min(len(MODEL.years), int(math.floor(current_index)))
    for boundary_index in range(1, completed + 1):
        year = MODEL.years[boundary_index - 1]
        if year in (2003, 2020):
            alpha, width = 238, 2
        elif year % 5 == 0 or boundary_index == completed:
            alpha, width = 205, 2
        else:
            alpha, width = 155, 1
        draw.line(closed(boundary_points(boundary_index, samples=540)), fill=NEUTRAL_RING + (alpha,), width=width)


def draw_growth_front(draw: ImageDraw.ImageDraw, current_index: float) -> None:
    if current_index >= len(MODEL.years):
        draw.line(closed(boundary_points(len(MODEL.years), samples=720)), fill=NEUTRAL_RING + (235,), width=2)
        return
    year_index = int(math.floor(current_index))
    frac = current_index - year_index
    points = []
    for i in range(720):
        angle = i * math.tau / 720
        inner = float(sample_periodic(MODEL.boundaries[year_index], angle))
        outer = float(sample_periodic(MODEL.boundaries[year_index + 1], angle))
        points.append(polar(inner + (outer - inner) * frac, angle))
    draw.line(closed(points), fill=NEUTRAL_RING + (230,), width=2)


def draw_pith(draw: ImageDraw.ImageDraw, frame_index: int) -> None:
    radius = PITH_RADIUS * smoothstep(min(1.0, frame_index / 24.0))
    if radius > 0:
        box = (TREE_CENTER[0] - radius, TREE_CENTER[1] - radius, TREE_CENTER[0] + radius, TREE_CENTER[1] + radius)
        draw.ellipse(box, fill=(231, 221, 200, 255), outline=(112, 105, 96, 145), width=1)


def draw_color_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.text((x, y), "weekly excess mortality palette", font=FONT_SMALL, fill=MID)
    legend_w = 330
    legend_h = 20
    for i in range(legend_w):
        excess = np.interp(i, [0, legend_w - 1], [PALETTE_VALUES[0], PALETTE_VALUES[-1]])
        color = tuple(int(v) for v in colors_for_excess(np.array([excess]))[0])
        draw.line((x + i, y + 32, x + i, y + 32 + legend_h), fill=color)
    draw.rectangle((x, y + 32, x + legend_w, y + 32 + legend_h), outline=(145, 145, 145), width=1)
    draw.text((x, y + 60), "-50%", font=FONT_TINY, fill=MID)
    draw.text((x + 143, y + 60), "0%", font=FONT_TINY, fill=MID)
    draw.text((x + 292, y + 60), "+120%", font=FONT_TINY, fill=MID)


def draw_year_rail(draw: ImageDraw.ImageDraw, x: int, top: int, bottom: int, idx_float: float) -> None:
    draw.line((x, top, x, bottom), fill=(174, 174, 174, 170), width=2)
    start = MODEL.years[0]
    end = MODEL.years[-1]
    for year in range(1995, 2030, 5):
        if year < start or year > end or year in (2020,):
            continue
        fraction = (year - start) / (end - start)
        y = top + fraction * (bottom - top)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(85, 85, 85, 210))
        draw.text((x + 18, y - 9), str(year), font=FONT_TINY, fill=MID)
    for event_year in (2003, 2020):
        fraction = (event_year - start) / (end - start)
        y = top + fraction * (bottom - top)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(35, 35, 35, 235))
        draw.text((x + 18, y - 10), str(event_year), font=FONT_SMALL, fill=INK)

    marker_y = top + min(1.0, max(0.0, idx_float / len(MODEL.years))) * (bottom - top)
    draw.rounded_rectangle((x - 20, marker_y - 10, x + 20, marker_y + 10), radius=10, fill=(54, 54, 54, 235))
    label_y = marker_y + 18 if idx_float > len(MODEL.years) - 3 else marker_y - 10
    draw.text((x + 56, label_y), "growth front", font=FONT_SMALL, fill=INK)


def draw_annotations(draw: ImageDraw.ImageDraw, frame_index: int, generated_count: int, new_cells: int) -> None:
    idx_float = current_year_index(frame_index)
    year_index = min(len(MODEL.years) - 1, int(math.floor(idx_float)))
    year = MODEL.years[year_index]
    annual_deaths = sum(day.deaths for day in MODEL.days_by_year[year_index])
    weighted_excess = np.average(
        [week.excess for week in MODEL.weeks_by_year[year_index]],
        weights=[max(1, week.deaths) for week in MODEL.weeks_by_year[year_index]],
    )

    panel_x = 1240
    draw.text((panel_x, 98), "Belgium mortality rings", font=FONT_TITLE, fill=INK)
    draw.text((panel_x, 143), "weekly signal, one death per cell", font=FONT_BODY, fill=MID)
    draw.text((panel_x, 210), f"{year}", font=FONT_SECTION, fill=INK)
    draw.text((panel_x + 92, 216), f"{annual_deaths:,} deaths", font=FONT_BODY, fill=INK)
    draw.text((panel_x, 248), f"{weighted_excess:+.1%} vs weekly baseline", font=FONT_BODY, fill=MID)

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
    note = "Angle = time within year; each annual band grows outward; weekly death counts create amplified local thickness."
    draw.text((82, 1024), note, font=FONT_SMALL, fill=(118, 118, 118))
    draw.text((1240, 1000), "Source: Statbel open data, Number of deaths per day, aggregated to calendar weeks.", font=FONT_TINY, fill=(130, 130, 130))


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

        frame = Image.fromarray(canvas).convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")
        draw_pith(draw, frame_index)
        idx_float = current_year_index(frame_index)
        draw_completed_boundaries(draw, idx_float)
        draw_growth_front(draw, idx_float)
        draw_annotations(draw, frame_index, generated_count, int(frame_counts[frame_index]))

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
    labels = [f"{seconds:g}s" for seconds in np.linspace(0, DURATION_SECONDS, len(frames))]
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
    print("weekly bumps: local ring thickness uses amplified weekly excess; counts remain one death per cell")
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
