from __future__ import annotations

import json
import math
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1440
HEIGHT = 1080
FPS = 24
DRAW_SECONDS = 36
REVEAL_SECONDS = 4
DURATION_SECONDS = DRAW_SECONDS + REVEAL_SECONDS
DRAW_FRAME_COUNT = DRAW_SECONDS * FPS
FRAME_COUNT = DURATION_SECONDS * FPS

ANGLE_SAMPLES = 1440
PITH_RADIUS = 20.0
MAX_RADIUS = 428.0
TREE_CENTER = (720, 548)
ANGLE_OFFSET = -math.pi / 2.0 + math.radians(13.0)
SEED = 20260613

BACKGROUND = (255, 255, 255)
NEUTRAL_RING = (236, 228, 210)
MID = (103, 101, 94)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "eurostat_country_rings"
SOURCE_DIR = OUTPUT_DIR / "sources"

EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "demo_r_mwk_ts?geo={geo}&sex=T&unit=NR&lang=en"
)


@dataclass(frozen=True, slots=True)
class CountrySpec:
    geo: str
    name: str
    slug: str
    palette_name: str
    palette: tuple[tuple[float, str], ...]


COUNTRIES = [
    CountrySpec(
        geo="NL",
        name="Netherlands",
        slug="netherlands",
        palette_name="Canal Ember",
        palette=(
            (-0.30, "#063B4A"),
            (-0.15, "#5D8792"),
            (0.00, "#F3E8C9"),
            (0.12, "#E7AF4F"),
            (0.30, "#C46234"),
            (0.60, "#7B1F2A"),
            (1.05, "#21060D"),
        ),
    ),
    CountrySpec(
        geo="SE",
        name="Sweden",
        slug="sweden",
        palette_name="Aurora Charcoal",
        palette=(
            (-0.30, "#143A5A"),
            (-0.15, "#4E9BA7"),
            (0.00, "#F0EAD2"),
            (0.12, "#B8D36C"),
            (0.30, "#D66C9F"),
            (0.60, "#68235B"),
            (1.05, "#110719"),
        ),
    ),
    CountrySpec(
        geo="FI",
        name="Finland",
        slug="finland",
        palette_name="Midnight Forest",
        palette=(
            (-0.30, "#0A2433"),
            (-0.15, "#3D6F72"),
            (0.00, "#EDE6CF"),
            (0.12, "#CDAA50"),
            (0.30, "#A75C32"),
            (0.60, "#56301D"),
            (1.05, "#090605"),
        ),
    ),
    CountrySpec(
        geo="CH",
        name="Switzerland",
        slug="switzerland",
        palette_name="Alpine Crimson",
        palette=(
            (-0.30, "#12324A"),
            (-0.15, "#7AA6AD"),
            (0.00, "#F7F1DE"),
            (0.12, "#E1B95F"),
            (0.30, "#D36A4A"),
            (0.60, "#8B1E2D"),
            (1.05, "#26040A"),
        ),
    ),
    CountrySpec(
        geo="AT",
        name="Austria",
        slug="austria",
        palette_name="Danube Garnet",
        palette=(
            (-0.30, "#183D4E"),
            (-0.15, "#80927B"),
            (0.00, "#EFE1BE"),
            (0.12, "#C99539"),
            (0.30, "#9E522B"),
            (0.60, "#5B1D30"),
            (1.05, "#16040C"),
        ),
    ),
]


@dataclass(slots=True)
class WeekRecord:
    year_index: int
    year: int
    iso_week: int
    week_count: int
    deaths: int
    baseline: float
    excess: float


@dataclass(slots=True)
class CountryModel:
    spec: CountrySpec
    years: list[int]
    weeks_by_year: list[list[WeekRecord]]
    boundaries: list[np.ndarray]
    total_deaths: int


@dataclass(slots=True)
class RectChunk:
    x: np.ndarray
    y: np.ndarray
    rgb: np.ndarray
    tx: np.ndarray
    ty: np.ndarray


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


FONT_SMALL = load_font(16)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    clean = value.strip().lstrip("#")
    return tuple(int(clean[index : index + 2], 16) for index in (0, 2, 4))


def colors_for_excess(excess: np.ndarray, palette: tuple[tuple[float, str], ...]) -> np.ndarray:
    values = np.array([stop for stop, _color in palette], dtype=float)
    colors = np.array([hex_to_rgb(color) for _stop, color in palette], dtype=float)
    clamped = np.clip(excess, values[0], values[-1])
    r = np.interp(clamped, values, colors[:, 0])
    g = np.interp(clamped, values, colors[:, 1])
    b = np.interp(clamped, values, colors[:, 2])
    return np.column_stack((r, g, b)).astype(np.float32)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


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


def sample_periodic(values: np.ndarray, angle: np.ndarray | float) -> np.ndarray | float:
    position = (np.asarray(angle) % math.tau) / math.tau * len(values)
    interpolated = circular_interp(values, position)
    if np.isscalar(angle):
        return float(interpolated)
    return interpolated


def closed(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points + points[:1]


def polar(radius: float, angle: float) -> tuple[float, float]:
    visual_angle = angle + ANGLE_OFFSET
    return (
        TREE_CENTER[0] + radius * math.cos(visual_angle),
        TREE_CENTER[1] + radius * math.sin(visual_angle),
    )


def eurostat_json(spec: CountrySpec) -> dict:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    cache = SOURCE_DIR / f"eurostat_demo_r_mwk_ts_{spec.geo}.json"
    if not cache.exists():
        url = EUROSTAT_URL.format(geo=spec.geo)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            cache.write_bytes(response.read())
    return json.loads(cache.read_text(encoding="utf-8"))


def load_weekly_dataframe(spec: CountrySpec) -> pd.DataFrame:
    data = eurostat_json(spec)
    time_index = data["dimension"]["time"]["category"]["index"]
    values = data.get("value", {})
    rows = []
    for time_label, index in sorted(time_index.items(), key=lambda item: item[1]):
        value = values.get(str(index))
        if value is None:
            continue
        if "-W" not in time_label:
            continue
        year_text, week_text = time_label.split("-W", maxsplit=1)
        year = int(year_text)
        iso_week = int(week_text)
        if 2000 <= year <= 2025:
            rows.append({"year": year, "iso_week": iso_week, "deaths": int(round(float(value)))})
    df = pd.DataFrame(rows).sort_values(["year", "iso_week"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No weekly data loaded for {spec.name}.")
    df["week_count"] = df.groupby("year")["iso_week"].transform("max").astype(int)
    return df


def compute_baseline(df: pd.DataFrame) -> pd.Series:
    baseline_years = df[df["year"].between(2015, 2019)]
    if baseline_years.empty:
        baseline_years = df[df["year"] < 2020]
    baseline_by_week = baseline_years.groupby("iso_week")["deaths"].median()
    global_median = float(baseline_years["deaths"].median())
    return df["iso_week"].map(baseline_by_week).fillna(global_median).clip(lower=1.0)


def build_boundaries(years: list[int], weeks_by_year: list[list[WeekRecord]]) -> list[np.ndarray]:
    annual_totals = np.array([sum(week.deaths for week in weeks) for weeks in weeks_by_year], dtype=float)
    mean_widths = (annual_totals / annual_totals.mean()) ** 0.35
    mean_widths = mean_widths / mean_widths.sum() * (MAX_RADIUS - PITH_RADIUS)
    boundaries = [np.full(ANGLE_SAMPLES, PITH_RADIUS, dtype=float)]

    for year_index, weeks in enumerate(weeks_by_year):
        local = np.ones(ANGLE_SAMPLES, dtype=float)
        weekly_rates = np.array([week.deaths for week in weeks], dtype=float)
        year_mean = float(np.mean(weekly_rates))
        for sample_index in range(ANGLE_SAMPLES):
            frac = sample_index / ANGLE_SAMPLES
            week = min(
                weeks,
                key=lambda candidate: min(
                    abs(((candidate.iso_week - 0.5) / candidate.week_count) - frac),
                    1.0 - abs(((candidate.iso_week - 0.5) / candidate.week_count) - frac),
                ),
            )
            factor = 1.0 + 0.74 * np.tanh((week.deaths / max(1.0, year_mean) - 1.0) / 0.22)
            local[sample_index] = factor
        local = circular_gaussian(local, sigma=10.0)
        local = np.clip(local, 0.48, 2.05)
        local /= local.mean()
        boundaries.append(boundaries[-1] + mean_widths[year_index] * local)
    return boundaries


def build_model(spec: CountrySpec) -> CountryModel:
    df = load_weekly_dataframe(spec)
    df["baseline"] = compute_baseline(df)
    df["excess"] = df["deaths"] / df["baseline"] - 1.0
    years = sorted(df["year"].unique().tolist())
    year_to_index = {year: index for index, year in enumerate(years)}
    weeks_by_year: list[list[WeekRecord]] = []
    for year in years:
        rows = df[df["year"].eq(year)]
        year_weeks = [
            WeekRecord(
                year_index=year_to_index[year],
                year=year,
                iso_week=int(row.iso_week),
                week_count=int(row.week_count),
                deaths=int(row.deaths),
                baseline=float(row.baseline),
                excess=float(row.excess),
            )
            for row in rows.itertuples(index=False)
        ]
        weeks_by_year.append(year_weeks)
    boundaries = build_boundaries(years, weeks_by_year)
    summary = df.copy()
    summary["country"] = spec.name
    summary["excess"] = summary["excess"].round(6)
    summary.to_csv(OUTPUT_DIR / f"{spec.slug}_weekly_deaths_2000_2025.csv", index=False)
    return CountryModel(
        spec=spec,
        years=years,
        weeks_by_year=weeks_by_year,
        boundaries=boundaries,
        total_deaths=int(df["deaths"].sum()),
    )


def radius_at(model: CountryModel, year_index: int, radial_t: np.ndarray, angle: np.ndarray) -> np.ndarray:
    inner = sample_periodic(model.boundaries[year_index], angle)
    outer = sample_periodic(model.boundaries[year_index + 1], angle)
    return inner + (outer - inner) * radial_t


def boundary_points(model: CountryModel, boundary_index: int, samples: int = 720) -> list[tuple[float, float]]:
    values = model.boundaries[boundary_index]
    phase = (boundary_index * 2.399963229728653) % math.tau
    return [
        polar(float(sample_periodic(values, phase + i * math.tau / samples)), phase + i * math.tau / samples)
        for i in range(samples)
    ]


def oriented_pixel_offsets(visual_angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tx = np.rint(-np.sin(visual_angle)).astype(np.int8)
    ty = np.rint(np.cos(visual_angle)).astype(np.int8)
    tx[(tx == 0) & (ty == 0)] = 1
    return tx, ty


def plot_rect_cells(canvas: np.ndarray, chunk: RectChunk) -> None:
    x = chunk.x
    y = chunk.y
    valid = (x >= 1) & (x < WIDTH - 2) & (y >= 1) & (y < HEIGHT - 2)
    if not np.any(valid):
        return
    xv = x[valid]
    yv = y[valid]
    rgb = chunk.rgb[valid]
    tx = chunk.tx[valid]
    ty = chunk.ty[valid]
    canvas[yv, xv] = rgb
    canvas[yv + ty, xv + tx] = rgb


def generate_frame_chunks(model: CountryModel) -> tuple[list[list[RectChunk]], np.ndarray]:
    chunks: list[list[RectChunk]] = [[] for _ in range(DRAW_FRAME_COUNT)]
    frame_counts = np.zeros(DRAW_FRAME_COUNT, dtype=np.int64)
    year_count = len(model.years)
    country_seed = sum((index + 1) * ord(char) for index, char in enumerate(model.spec.geo))

    for year_index, weeks in enumerate(model.weeks_by_year):
        max_week_count = max(week.week_count for week in weeks)
        excess_grid = np.zeros(max_week_count, dtype=float)
        for week in weeks:
            excess_grid[week.iso_week - 1] = week.excess
        excess_grid = circular_gaussian(excess_grid, sigma=0.62)

        for week in weeks:
            n = week.deaths
            rng = np.random.default_rng(SEED + country_seed + week.year * 10_000 + week.iso_week)
            start_frac = (week.iso_week - 1.0) / week.week_count
            end_frac = week.iso_week / week.week_count
            theta = rng.uniform(math.tau * start_frac, math.tau * end_frac, n)
            theta += rng.normal(0.0, math.tau * rng.uniform(0.05, 0.14) / week.week_count, n)
            radial_t = rng.random(n)
            theta += (radial_t - 0.5) * rng.normal(0.0, math.tau * 0.28 / week.week_count)

            radius = radius_at(model, year_index, radial_t, theta) + rng.normal(0.0, 0.08, n)
            visual_angle = theta + ANGLE_OFFSET
            x = np.rint(TREE_CENTER[0] + radius * np.cos(visual_angle) + rng.normal(0.0, 0.16, n)).astype(np.int32)
            y = np.rint(TREE_CENTER[1] + radius * np.sin(visual_angle) + rng.normal(0.0, 0.16, n)).astype(np.int32)

            week_float = (theta % math.tau) / math.tau * max_week_count
            excess = circular_interp(excess_grid, week_float)
            base_color = colors_for_excess(excess, model.spec.palette)
            rgb = np.clip(base_color + rng.normal(0.0, 2.6, (n, 3)), 0, 255).astype(np.uint8)
            tx, ty = oriented_pixel_offsets(visual_angle)

            radial_progress = np.clip(radial_t + rng.normal(0.0, 0.014, n), 0.0, 0.998)
            progress = (year_index + radial_progress) / year_count
            birth_frames = np.clip((progress * (DRAW_FRAME_COUNT - 1)).astype(np.int32), 0, DRAW_FRAME_COUNT - 1)
            for frame_index in np.unique(birth_frames):
                mask = birth_frames == frame_index
                chunks[int(frame_index)].append(RectChunk(x=x[mask], y=y[mask], rgb=rgb[mask], tx=tx[mask], ty=ty[mask]))
                frame_counts[int(frame_index)] += int(mask.sum())

        if year_index % 5 == 0:
            print(f"{model.spec.name}: prepared cells through {model.years[year_index]}")
    return chunks, frame_counts


def current_year_index(model: CountryModel, frame_index: int) -> float:
    if frame_index >= DRAW_FRAME_COUNT:
        return float(len(model.years))
    return min(len(model.years), frame_index / max(1, DRAW_FRAME_COUNT - 1) * len(model.years))


def draw_completed_boundaries(draw: ImageDraw.ImageDraw, model: CountryModel, current_index: float) -> None:
    completed = min(len(model.years), int(math.floor(current_index)))
    for boundary_index in range(1, completed + 1):
        year = model.years[boundary_index - 1]
        if year in (2003, 2015, 2020, 2022):
            alpha, width = 214, 2
        elif year % 5 == 0 or boundary_index == completed:
            alpha, width = 176, 2
        else:
            alpha, width = 116, 1
        draw.line(closed(boundary_points(model, boundary_index, samples=560)), fill=NEUTRAL_RING + (alpha,), width=width)


def draw_growth_front(draw: ImageDraw.ImageDraw, model: CountryModel, current_index: float) -> None:
    if current_index >= len(model.years):
        draw.line(closed(boundary_points(model, len(model.years), samples=760)), fill=NEUTRAL_RING + (235,), width=2)
        return
    year_index = int(math.floor(current_index))
    frac = current_index - year_index
    points = []
    for i in range(720):
        angle = i * math.tau / 720
        inner = float(sample_periodic(model.boundaries[year_index], angle))
        outer = float(sample_periodic(model.boundaries[year_index + 1], angle))
        points.append(polar(inner + (outer - inner) * frac, angle))
    draw.line(closed(points), fill=NEUTRAL_RING + (226,), width=2)


def draw_pith(draw: ImageDraw.ImageDraw, frame_index: int) -> None:
    radius = PITH_RADIUS * smoothstep(min(1.0, frame_index / 18.0))
    if radius <= 0:
        return
    box = (
        TREE_CENTER[0] - radius,
        TREE_CENTER[1] - radius,
        TREE_CENTER[0] + radius,
        TREE_CENTER[1] + radius,
    )
    draw.ellipse(box, fill=NEUTRAL_RING + (255,), outline=(112, 105, 96, 145), width=1)


def output_paths(model: CountryModel) -> dict[str, Path]:
    prefix = f"{model.spec.slug}_mortality_rings_eurostat_2000_2025"
    return {
        "mp4": OUTPUT_DIR / f"{prefix}.mp4",
        "x_mp4": OUTPUT_DIR / f"{prefix}_x_h264_aac.mp4",
        "final_frame": OUTPUT_DIR / f"{prefix}_final_frame.png",
        "contact_sheet": OUTPUT_DIR / f"{prefix}_contact_sheet.png",
    }


def render_all_frames(model: CountryModel, chunks: list[list[RectChunk]], frame_counts: np.ndarray) -> tuple[Image.Image, list[Image.Image]]:
    paths = output_paths(model)
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    final_frame: Image.Image | None = None
    contact_frames: list[Image.Image] = []
    contact_indices = {0, DRAW_FRAME_COUNT // 4, DRAW_FRAME_COUNT // 2, DRAW_FRAME_COUNT * 3 // 4, FRAME_COUNT - 1}
    generated_count = 0

    writer = cv2.VideoWriter(str(paths["mp4"]), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer.")

    for frame_index in range(FRAME_COUNT):
        if frame_index < DRAW_FRAME_COUNT:
            for chunk in chunks[frame_index]:
                plot_rect_cells(canvas, chunk)
            generated_count += int(frame_counts[frame_index])

        frame = Image.fromarray(canvas).convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")
        idx_float = current_year_index(model, frame_index)
        draw_pith(draw, min(frame_index, DRAW_FRAME_COUNT - 1))
        draw_completed_boundaries(draw, model, idx_float)
        draw_growth_front(draw, model, idx_float)

        rgb = frame.convert("RGB")
        writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        final_frame = rgb
        if frame_index in contact_indices:
            contact_frames.append(rgb.copy())
        if frame_index % max(1, FRAME_COUNT // 10) == 0:
            print(f"{model.spec.name}: rendered {frame_index:04d}/{FRAME_COUNT} cells={generated_count:,}/{model.total_deaths:,}")

    writer.release()
    if final_frame is None:
        raise RuntimeError("No frames rendered.")
    return final_frame, contact_frames


def save_contact_sheet(model: CountryModel, frames: list[Image.Image]) -> None:
    if not frames:
        return
    paths = output_paths(model)
    thumb_w = 288
    thumb_h = 216
    padding = 18
    label_h = 24
    sheet = Image.new(
        "RGB",
        (thumb_w * len(frames) + padding * (len(frames) + 1), thumb_h + padding * 2 + label_h),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    labels = [f"{seconds:g}s" for seconds in [0, DRAW_SECONDS / 4, DRAW_SECONDS / 2, DRAW_SECONDS * 3 / 4, DURATION_SECONDS]]
    for index, frame in enumerate(frames):
        x = padding + index * (thumb_w + padding)
        y = padding
        sheet.paste(frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + thumb_h + 6), labels[index], font=FONT_SMALL, fill=MID)
    sheet.save(paths["contact_sheet"])


def encode_x_compatible(model: CountryModel) -> None:
    paths = output_paths(model)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(paths["mp4"]),
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
        str(paths["x_mp4"]),
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


def render_country(spec: CountrySpec) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(spec)
    paths = output_paths(model)
    print()
    print(f"{model.spec.name}: {model.years[0]}-{model.years[-1]} ({len(model.years)} rings)")
    print(f"{model.spec.name}: total cells/deaths {model.total_deaths:,}")
    print(f"{model.spec.name}: palette {model.spec.palette_name}")
    chunks, frame_counts = generate_frame_chunks(model)
    final_frame, contact_frames = render_all_frames(model, chunks, frame_counts)
    final_frame.save(paths["final_frame"])
    save_contact_sheet(model, contact_frames)
    encode_x_compatible(model)
    print(f"{model.spec.name}: video {paths['x_mp4']} ({paths['x_mp4'].stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"{model.spec.name}: final frame {paths['final_frame']}")
    print(f"{model.spec.name}: contact sheet {paths['contact_sheet']}")
    for line in inspect_video(paths["x_mp4"]):
        print(f"  {line}")


def main() -> None:
    for spec in COUNTRIES:
        render_country(spec)


if __name__ == "__main__":
    main()
