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

SEED = 20240611
ANGLE_SAMPLES = 1440
TARGET_CELLS = 100_000

TREE_CENTER = (650, 540)
PITH_RADIUS = 14.0
MAX_RADIUS = 445.0
ANGLE_OFFSET = -math.pi / 2.0

BACKGROUND = (255, 255, 255)
INK = (35, 35, 35)
MID = (105, 105, 105)
LIGHT = (190, 190, 190)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
QUERY_FILE = Path(r"C:\Users\Florent\Downloads\pxapi-api_table_px-x-0102020206_111.px.json")
PXWEB_URL = "https://www.pxweb.bfs.admin.ch/api/v1/en/px-x-0102020206_111/px-x-0102020206_111.px"

RAW_JSON_PATH = OUTPUT_DIR / "bfs_monthly_deaths_jsonstat.json"
MONTHLY_CSV_PATH = OUTPUT_DIR / "bfs_monthly_deaths_derived.csv"
MP4_PATH = OUTPUT_DIR / "swiss_mortality_tree_1877_2024_30s.mp4"
X_MP4_PATH = OUTPUT_DIR / "swiss_mortality_tree_1877_2024_x_h264_aac.mp4"
FINAL_FRAME_PATH = OUTPUT_DIR / "swiss_mortality_tree_1877_2024_final_frame.png"
CONTACT_SHEET_PATH = OUTPUT_DIR / "swiss_mortality_tree_1877_2024_contact_sheet.png"


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
class MonthRecord:
    year: int
    month: int
    deaths: int
    baseline: float
    excess: float


@dataclass(slots=True)
class DayRecord:
    year_index: int
    year: int
    month: int
    day: int
    day_of_year: int
    days_in_year: int
    deaths: int
    baseline_daily: float
    excess: float
    angle0: float
    angle1: float


@dataclass(slots=True)
class MortalityModel:
    years: list[int]
    month_records: list[MonthRecord]
    days_by_year: list[list[DayRecord]]
    boundaries: list[np.ndarray]
    year_mean_widths: np.ndarray
    deaths_per_cell: int
    total_deaths: int
    total_cells: int
    source: str


@dataclass(slots=True)
class Cell:
    year_index: int
    theta0: float
    theta1: float
    t0: float
    t1: float
    birth_frame: int
    mature_frame: int
    fill: tuple[int, int, int, int]
    outline: tuple[int, int, int, int]
    active_outline: tuple[int, int, int, int]
    excess: float
    deaths_represented: int
    wobble: float


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


def smooth_circular(values: np.ndarray, passes: int = 3) -> np.ndarray:
    smoothed = values.astype(float).copy()
    kernel = np.array([0.08, 0.20, 0.44, 0.20, 0.08])
    offsets = np.array([-2, -1, 0, 1, 2])
    for _ in range(passes):
        next_values = np.zeros_like(smoothed)
        for weight, offset in zip(kernel, offsets):
            next_values += weight * np.roll(smoothed, offset)
        smoothed = next_values
    return smoothed


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
    source = str(dataset.get("source", "FSO"))
    return complete_years, complete_monthly, source


def compute_monthly_baseline(monthly: np.ndarray) -> np.ndarray:
    baseline = np.zeros_like(monthly, dtype=float)
    for year_index in range(monthly.shape[0]):
        for month_index in range(12):
            start = max(0, year_index - 5)
            previous = monthly[start:year_index, month_index]
            if len(previous) >= 3:
                baseline[year_index, month_index] = float(np.mean(previous))
            else:
                end = min(monthly.shape[0], year_index + 6)
                baseline[year_index, month_index] = float(np.mean(monthly[0:end, month_index]))
    return np.maximum(baseline, 1.0)


def distribute_integer(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(weights.astype(float), 1e-9)
    shares = weights / weights.sum() * total
    values = np.floor(shares).astype(int)
    remainder = int(total - values.sum())
    if remainder > 0:
        order = np.argsort(-(shares - values))
        values[order[:remainder]] += 1
    return values


def distribute_month_to_days(year: int, month: int, total: int, excess: float) -> np.ndarray:
    days = calendar.monthrange(year, month)[1]
    rng = np.random.default_rng(SEED + year * 100 + month)
    x = np.arange(days)
    weights = rng.lognormal(mean=0.0, sigma=0.26, size=days)

    weekly_phase = rng.uniform(0.0, math.tau)
    weights *= 1.0 + 0.05 * np.sin(x * math.tau / 7.0 + weekly_phase)

    if excess > 0.12:
        center = rng.uniform(0.20, 0.80) * max(1, days - 1)
        width = rng.uniform(2.0, max(3.0, days * 0.22))
        pulse = np.exp(-0.5 * ((x - center) / width) ** 2)
        weights *= 1.0 + min(0.95, excess * 1.25) * pulse

    if month in (1, 2, 12):
        winter_wave = 1.0 + 0.08 * np.cos((x / max(1, days - 1)) * math.pi)
        weights *= winter_wave

    for _ in range(2):
        padded = np.r_[weights[0], weights, weights[-1]]
        weights = 0.22 * padded[:-2] + 0.56 * padded[1:-1] + 0.22 * padded[2:]

    return distribute_integer(total, weights)


def excess_to_fill(excess: float) -> tuple[int, int, int, int]:
    clamped = max(-0.35, min(0.85, excess))
    if clamped >= 0:
        shade = int(np.interp(clamped, [0.0, 0.85], [218, 70]))
    else:
        shade = int(np.interp(clamped, [-0.35, 0.0], [250, 218]))
    return (shade, shade, shade, 255)


def build_mortality_model() -> MortalityModel:
    data = fetch_pxweb_data()
    years, monthly, source = parse_monthly_records(data)
    baseline = compute_monthly_baseline(monthly)
    month_records: list[MonthRecord] = []
    days_by_year: list[list[DayRecord]] = []

    for year_index, year in enumerate(years):
        year_days: list[DayRecord] = []
        day_of_year = 0
        for month in range(1, 13):
            deaths = int(monthly[year_index, month - 1])
            expected = float(baseline[year_index, month - 1])
            excess = deaths / expected - 1.0
            month_records.append(MonthRecord(year, month, deaths, expected, excess))

            daily_deaths = distribute_month_to_days(year, month, deaths, excess)
            days_in_month = len(daily_deaths)
            days_in_year = 366 if calendar.isleap(year) else 365
            baseline_daily = expected / days_in_month

            for day in range(1, days_in_month + 1):
                angle0 = math.tau * day_of_year / days_in_year
                angle1 = math.tau * (day_of_year + 1) / days_in_year
                year_days.append(
                    DayRecord(
                        year_index=year_index,
                        year=year,
                        month=month,
                        day=day,
                        day_of_year=day_of_year,
                        days_in_year=days_in_year,
                        deaths=int(daily_deaths[day - 1]),
                        baseline_daily=baseline_daily,
                        excess=excess,
                        angle0=angle0,
                        angle1=angle1,
                    )
                )
                day_of_year += 1
        days_by_year.append(year_days)

    total_deaths = int(monthly.sum())
    deaths_per_cell = max(1, int(math.ceil(total_deaths / TARGET_CELLS)))

    annual_deaths = monthly.sum(axis=1).astype(float)
    raw_year_widths = (annual_deaths / annual_deaths.mean()) ** 0.52
    year_mean_widths = raw_year_widths / raw_year_widths.sum() * (MAX_RADIUS - PITH_RADIUS)

    boundaries = build_boundaries(days_by_year, year_mean_widths)
    total_cells = estimate_cell_count(days_by_year, deaths_per_cell)
    write_monthly_csv(years, monthly, baseline)

    return MortalityModel(
        years=years,
        month_records=month_records,
        days_by_year=days_by_year,
        boundaries=boundaries,
        year_mean_widths=year_mean_widths,
        deaths_per_cell=deaths_per_cell,
        total_deaths=total_deaths,
        total_cells=total_cells,
        source=source,
    )


def build_boundaries(days_by_year: list[list[DayRecord]], year_mean_widths: np.ndarray) -> list[np.ndarray]:
    boundaries = [np.full(ANGLE_SAMPLES, PITH_RADIUS, dtype=float)]

    for year_index, days in enumerate(days_by_year):
        daily = np.array([max(1, day.deaths) for day in days], dtype=float)
        annual_average = daily.mean()
        local = np.zeros(ANGLE_SAMPLES, dtype=float)
        for sample_index in range(ANGLE_SAMPLES):
            angle_fraction = sample_index / ANGLE_SAMPLES
            day_index = min(len(days) - 1, int(angle_fraction * len(days)))
            local[sample_index] = (daily[day_index] / annual_average) ** 0.52

        local = smooth_circular(local, passes=10)
        local = np.clip(local, 0.58, 1.92)
        local /= local.mean()
        width = year_mean_widths[year_index] * local
        boundaries.append(boundaries[-1] + width)

    return boundaries


def estimate_cell_count(days_by_year: list[list[DayRecord]], deaths_per_cell: int) -> int:
    total = 0
    for days in days_by_year:
        for day in days:
            total += max(1, int(round(day.deaths / deaths_per_cell)))
    return total


def write_monthly_csv(years: list[int], monthly: np.ndarray, baseline: np.ndarray) -> None:
    lines = ["year,month,deaths,baseline,excess\n"]
    for year_index, year in enumerate(years):
        for month in range(1, 13):
            deaths = int(monthly[year_index, month - 1])
            expected = float(baseline[year_index, month - 1])
            excess = deaths / expected - 1.0
            lines.append(f"{year},{month},{deaths},{expected:.3f},{excess:.6f}\n")
    MONTHLY_CSV_PATH.write_text("".join(lines), encoding="utf-8")


MODEL = build_mortality_model()


def sample_periodic(values: np.ndarray, angle: float) -> float:
    position = (angle % math.tau) / math.tau * len(values)
    left = int(math.floor(position)) % len(values)
    right = (left + 1) % len(values)
    frac = position - math.floor(position)
    return float(values[left] * (1.0 - frac) + values[right] * frac)


def radius_at(year_index: int, radial_t: float, angle: float) -> float:
    if year_index < 0:
        return sample_periodic(MODEL.boundaries[0], angle) * radial_t
    inner = sample_periodic(MODEL.boundaries[year_index], angle)
    outer = sample_periodic(MODEL.boundaries[year_index + 1], angle)
    return inner + (outer - inner) * radial_t


def boundary_points(year_index: int, samples: int = 720) -> list[tuple[float, float]]:
    values = MODEL.boundaries[year_index]
    return [polar(sample_periodic(values, i * math.tau / samples), i * math.tau / samples) for i in range(samples)]


def make_cell_polygon(cell: Cell, growth: float = 1.0) -> list[tuple[float, float]]:
    growth = smoothstep(growth)
    current_t1 = cell.t0 + (cell.t1 - cell.t0) * growth
    theta0 = cell.theta0
    theta1 = cell.theta1
    theta_mid = (theta0 + theta1) * 0.5
    t_mid = (cell.t0 + current_t1) * 0.5
    wobble = cell.wobble
    angular_drift = max(-0.0022, min(0.0022, wobble * 0.010))

    def point(t_value: float, angle: float, offset: float = 0.0) -> tuple[float, float]:
        span = max(1e-6, cell.t1 - cell.t0)
        drifted_angle = angle + angular_drift * ((t_value - cell.t0) / span)
        return polar(radius_at(cell.year_index, t_value, drifted_angle) + offset, drifted_angle)

    return [
        point(cell.t0, theta0, -0.18 * wobble),
        point(cell.t0, theta_mid, -0.08 * wobble),
        point(cell.t0, theta1, 0.10 * wobble),
        point(t_mid, theta1, 0.18 * wobble),
        point(current_t1, theta1, 0.22 * wobble),
        point(current_t1, theta_mid, 0.08 * wobble),
        point(current_t1, theta0, -0.18 * wobble),
        point(t_mid, theta0, -0.22 * wobble),
    ]


def draw_cell(draw: ImageDraw.ImageDraw, cell: Cell, growth: float = 1.0, active: bool = False) -> None:
    fill = cell.fill
    outline = cell.active_outline if active else cell.outline
    if active:
        alpha = int(100 + 155 * smoothstep(growth))
        fill = (fill[0], fill[1], fill[2], alpha)
    draw.polygon(make_cell_polygon(cell, growth), fill=fill, outline=outline)


def generate_pith_cells() -> list[Cell]:
    rng = np.random.default_rng(SEED + 7)
    cells: list[Cell] = []
    rings = [(0.00, 0.35, 12), (0.35, 0.70, 20), (0.70, 1.00, 28)]
    for ring_index, (t0, t1, count) in enumerate(rings):
        widths = rng.lognormal(0.0, 0.22, count)
        edges = np.r_[0.0, np.cumsum(widths / widths.sum() * math.tau)]
        phase = rng.uniform(0.0, math.tau)
        for index in range(count):
            birth_frame = int(rng.integers(0, 20 + ring_index * 12))
            cells.append(
                Cell(
                    year_index=-1,
                    theta0=float(edges[index] + phase),
                    theta1=float(edges[index + 1] + phase),
                    t0=t0,
                    t1=t1,
                    birth_frame=birth_frame,
                    mature_frame=birth_frame + int(rng.integers(3, 8)),
                    fill=(235, 235, 235, 255),
                    outline=(145, 145, 145, 118),
                    active_outline=(55, 55, 55, 210),
                    excess=0.0,
                    deaths_represented=0,
                    wobble=float(rng.normal(0.0, 0.35)),
                )
            )
    return cells


def generate_cells() -> list[Cell]:
    rng = np.random.default_rng(SEED + 100)
    cells = generate_pith_cells()

    for year_index, days in enumerate(MODEL.days_by_year):
        for day in days:
            cell_count = max(1, int(round(day.deaths / MODEL.deaths_per_cell)))
            local_width = sample_periodic(MODEL.boundaries[year_index + 1], (day.angle0 + day.angle1) * 0.5) - sample_periodic(
                MODEL.boundaries[year_index],
                (day.angle0 + day.angle1) * 0.5,
            )
            radial_layers = max(1, min(4, int(round(local_width / 1.8))))
            columns = max(1, int(math.ceil(cell_count / radial_layers)))
            day_span = day.angle1 - day.angle0
            base_fill = excess_to_fill(day.excess)
            outline_shade = max(55, min(190, base_fill[0] - 35))
            active_shade = max(35, min(120, base_fill[0] - 90))

            for cell_index in range(cell_count):
                layer = cell_index % radial_layers
                column = cell_index // radial_layers
                theta0 = day.angle0 + day_span * column / columns
                theta1 = day.angle0 + day_span * (column + 1) / columns
                theta_jitter = day_span / columns * rng.uniform(-0.18, 0.18)
                theta0 += theta_jitter
                theta1 += theta_jitter
                t0 = max(0.0, layer / radial_layers + rng.uniform(-0.018, 0.006))
                t1 = min(1.0, (layer + 1) / radial_layers + rng.uniform(-0.006, 0.018))
                if t1 <= t0:
                    t1 = min(1.0, t0 + 0.04)

                progress = (year_index + (day.day_of_year + column / max(1, columns)) / day.days_in_year) / len(MODEL.years)
                birth_frame = int(progress * (FRAME_COUNT - 1))
                mature_frame = min(FRAME_COUNT - 1, birth_frame + int(rng.integers(2, 7)))
                represented = MODEL.deaths_per_cell
                if cell_index == cell_count - 1:
                    represented = max(1, day.deaths - MODEL.deaths_per_cell * (cell_count - 1))

                cells.append(
                    Cell(
                        year_index=year_index,
                        theta0=float(theta0),
                        theta1=float(theta1),
                        t0=float(t0),
                        t1=float(t1),
                        birth_frame=birth_frame,
                        mature_frame=mature_frame,
                        fill=base_fill,
                        outline=(outline_shade, outline_shade, outline_shade, 68),
                        active_outline=(active_shade, active_shade, active_shade, 220),
                        excess=day.excess,
                        deaths_represented=int(represented),
                        wobble=float(rng.normal(0.0, 0.18)),
                    )
                )

    cells.sort(key=lambda item: (item.birth_frame, item.mature_frame))
    return cells


CELLS = generate_cells()


def current_year_index(frame_index: int) -> float:
    return min(len(MODEL.years), frame_index / max(1, FRAME_COUNT - 1) * len(MODEL.years))


def draw_completed_boundaries(draw: ImageDraw.ImageDraw, current_index: float) -> None:
    completed = min(len(MODEL.years), int(math.floor(current_index)))
    for boundary_index in range(0, completed + 1):
        if boundary_index == 0:
            continue
        year = MODEL.years[boundary_index - 1]
        if year % 10 == 0 or boundary_index == completed or year in (1918, 2020):
            shade = 46 if year in (1918, 2020) else 92
            alpha = 185 if year in (1918, 2020) else 120
            width = 2 if year in (1918, 2020) else 1
        else:
            shade = 138
            alpha = 42
            width = 1
        draw.line(closed(boundary_points(boundary_index, samples=540)), fill=(shade, shade, shade, alpha), width=width)


def draw_growth_front(draw: ImageDraw.ImageDraw, current_index: float) -> None:
    if current_index >= len(MODEL.years):
        draw.line(closed(boundary_points(len(MODEL.years), samples=720)), fill=(48, 48, 48, 210), width=2)
        return

    year_index = int(math.floor(current_index))
    frac = current_index - year_index
    points = [polar(radius_at(year_index, frac, i * math.tau / 720), i * math.tau / 720) for i in range(720)]
    draw.line(closed(points), fill=(45, 45, 45, 185), width=2)


def draw_month_ticks(draw: ImageDraw.ImageDraw) -> None:
    outer = MODEL.boundaries[-1]
    labels = [(1, "Jan"), (4, "Apr"), (7, "Jul"), (10, "Oct")]
    days_in_reference = 365
    starts = [0]
    total = 0
    for month in range(1, 13):
        total += calendar.monthrange(2021, month)[1]
        starts.append(total)

    for month, label in labels:
        angle = math.tau * starts[month - 1] / days_in_reference
        radius = sample_periodic(outer, angle)
        p1 = polar(radius + 8, angle)
        p2 = polar(radius + 28, angle)
        draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(120, 120, 120, 130), width=1)
        text = polar(radius + 40, angle)
        label_x = max(36, min(WIDTH - 80, text[0] - 14))
        label_y = max(28, min(HEIGHT - 80, text[1] - 9))
        draw.text((label_x, label_y), label, font=FONT_TINY, fill=MID)


def draw_annotations(draw: ImageDraw.ImageDraw, frame_index: int, generated_count: int, active_count: int) -> None:
    idx_float = current_year_index(frame_index)
    year_index = min(len(MODEL.years) - 1, int(math.floor(idx_float)))
    year = MODEL.years[year_index]
    annual_deaths = sum(day.deaths for day in MODEL.days_by_year[year_index])
    year_excess = np.mean([record.excess for record in MODEL.month_records if record.year == year])
    excess_label = f"{year_excess:+.1%} vs baseline"

    panel_x = 1240
    draw.text((panel_x, 98), "Swiss mortality rings", font=FONT_TITLE, fill=INK)
    draw.text((panel_x, 143), "monthly deaths mapped as tree-like annual growth", font=FONT_BODY, fill=MID)

    draw.text((panel_x, 210), f"{year}", font=FONT_SECTION, fill=INK)
    draw.text((panel_x + 92, 216), f"{annual_deaths:,} deaths", font=FONT_BODY, fill=INK)
    draw.text((panel_x, 248), excess_label, font=FONT_BODY, fill=MID)

    stat_y = 318
    draw.text((panel_x, stat_y), "cells generated", font=FONT_SMALL, fill=MID)
    draw.text((panel_x + 178, stat_y - 7), f"{generated_count:,} / {len(CELLS):,}", font=FONT_SECTION, fill=INK)
    draw.text((panel_x, stat_y + 42), "scale", font=FONT_SMALL, fill=MID)
    draw.text((panel_x + 178, stat_y + 35), f"1 cell = {MODEL.deaths_per_cell} deaths", font=FONT_BODY, fill=INK)
    draw.text((panel_x, stat_y + 78), "currently growing", font=FONT_SMALL, fill=MID)
    draw.text((panel_x + 178, stat_y + 71), f"{active_count:,}", font=FONT_BODY, fill=INK)

    draw_color_legend(draw, panel_x, 475)
    draw_year_rail(draw, panel_x + 36, 645, 920, idx_float)

    draw.line((82, 1000, 1098, 1000), fill=(214, 214, 214, 200), width=1)
    note = "Angle = time within year; ring thickness = deaths; shade = excess mortality versus trailing same-month baseline."
    draw.text((82, 1024), note, font=FONT_SMALL, fill=(118, 118, 118))
    source = "Source: Swiss Federal Statistical Office PXWEB, deaths per month and mortality since 1803."
    draw.text((1240, 1000), source, font=FONT_TINY, fill=(130, 130, 130))


def draw_color_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.text((x, y), "excess mortality shade", font=FONT_SMALL, fill=MID)
    legend_w = 280
    legend_h = 18
    for i in range(legend_w):
        excess = np.interp(i, [0, legend_w - 1], [-0.35, 0.85])
        color = excess_to_fill(float(excess))[:3]
        draw.line((x + i, y + 30, x + i, y + 30 + legend_h), fill=color)
    draw.rectangle((x, y + 30, x + legend_w, y + 30 + legend_h), outline=(150, 150, 150), width=1)
    draw.text((x, y + 55), "below", font=FONT_TINY, fill=MID)
    draw.text((x + 118, y + 55), "expected", font=FONT_TINY, fill=MID)
    draw.text((x + 238, y + 55), "above", font=FONT_TINY, fill=MID)


def draw_year_rail(draw: ImageDraw.ImageDraw, x: int, top: int, bottom: int, idx_float: float) -> None:
    draw.line((x, top, x, bottom), fill=(174, 174, 174, 180), width=2)
    start = MODEL.years[0]
    end = MODEL.years[-1]
    for year in range(1880, 2030, 20):
        if year < start or year > end:
            continue
        if any(abs(year - event_year) <= 2 for event_year in (1918, 2020)):
            continue
        fraction = (year - start) / (end - start)
        y = top + fraction * (bottom - top)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(85, 85, 85, 220))
        draw.text((x + 18, y - 9), str(year), font=FONT_TINY, fill=MID)

    for event_year in (1918, 2020):
        if start <= event_year <= end:
            fraction = (event_year - start) / (end - start)
            y = top + fraction * (bottom - top)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(35, 35, 35, 235))
            label_y = y - 28 if event_year == 2020 else y - 10
            draw.text((x + 18, label_y), str(event_year), font=FONT_SMALL, fill=INK)

    marker_y = top + min(1.0, max(0.0, idx_float / len(MODEL.years))) * (bottom - top)
    draw.rounded_rectangle((x - 20, marker_y - 10, x + 20, marker_y + 10), radius=10, fill=(54, 54, 54, 235))
    marker_label_y = marker_y + 18 if idx_float > len(MODEL.years) - 5 else marker_y - 10
    draw.text((x + 56, marker_label_y), "growth front", font=FONT_SMALL, fill=INK)


def render_all_frames() -> tuple[Image.Image, list[Image.Image]]:
    permanent_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    birth_cursor = 0
    active_cells: list[Cell] = []
    final_frame: Image.Image | None = None
    contact_frames: list[Image.Image] = []
    contact_indices = {0, FRAME_COUNT // 4, FRAME_COUNT // 2, FRAME_COUNT * 3 // 4, FRAME_COUNT - 1}

    writer = cv2.VideoWriter(str(MP4_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer.")

    for frame_index in range(FRAME_COUNT):
        while birth_cursor < len(CELLS) and CELLS[birth_cursor].birth_frame <= frame_index:
            active_cells.append(CELLS[birth_cursor])
            birth_cursor += 1

        permanent_draw = ImageDraw.Draw(permanent_layer, "RGBA")
        still_active: list[Cell] = []
        for cell in active_cells:
            if cell.mature_frame <= frame_index:
                draw_cell(permanent_draw, cell, growth=1.0, active=False)
            else:
                still_active.append(cell)
        active_cells = still_active

        frame = Image.new("RGBA", (WIDTH, HEIGHT), BACKGROUND + (255,))
        frame.alpha_composite(permanent_layer)
        draw = ImageDraw.Draw(frame, "RGBA")

        for cell in active_cells:
            growth = (frame_index - cell.birth_frame) / max(1, cell.mature_frame - cell.birth_frame)
            draw_cell(draw, cell, growth=growth, active=True)

        idx_float = current_year_index(frame_index)
        draw_completed_boundaries(draw, idx_float)
        draw_growth_front(draw, idx_float)
        draw_month_ticks(draw)
        draw_annotations(draw, frame_index, generated_count=birth_cursor, active_count=len(active_cells))

        rgb = frame.convert("RGB")
        writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        final_frame = rgb
        if frame_index in contact_indices:
            contact_frames.append(rgb.copy())

        if frame_index % max(1, FRAME_COUNT // 10) == 0:
            print(f"rendered {frame_index:04d}/{FRAME_COUNT} cells={birth_cursor:,}/{len(CELLS):,} active={len(active_cells):,}")

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
        "28",
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
    print(f"total deaths: {MODEL.total_deaths:,}")
    print(f"scale: 1 cell = {MODEL.deaths_per_cell} deaths")
    print(f"generated cells: {len(CELLS):,}")
    final_frame, contact_frames = render_all_frames()
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
