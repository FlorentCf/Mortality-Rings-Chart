from __future__ import annotations

import csv
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
DRAW_SECONDS = 44
REVEAL_SECONDS = 4
DURATION_SECONDS = DRAW_SECONDS + REVEAL_SECONDS
DRAW_FRAME_COUNT = DRAW_SECONDS * FPS
FRAME_COUNT = DURATION_SECONDS * FPS

ANGLE_SAMPLES = 1440
PITH_RADIUS = 18.0
MAX_RADIUS = 430.0
TREE_CENTER = (720, 548)
ANGLE_OFFSET = -math.pi / 2.0 + math.radians(11.0)
SEED = 20260614

BACKGROUND = (255, 255, 255)
NEUTRAL_RING = (233, 224, 204)
MID = (98, 94, 83)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "hyde_land_use_rings"
SOURCE_DIR = OUTPUT_DIR / "sources"
CSV_PATH = OUTPUT_DIR / "hyde35_global_land_use_signals.csv"
MP4_PATH = OUTPUT_DIR / "hyde35_land_use_rings_10000BCE_2025.mp4"
X_MP4_PATH = OUTPUT_DIR / "hyde35_land_use_rings_10000BCE_2025_x_h264_aac.mp4"
FINAL_FRAME_PATH = OUTPUT_DIR / "hyde35_land_use_rings_10000BCE_2025_final_frame.png"
CONTACT_SHEET_PATH = OUTPUT_DIR / "hyde35_land_use_rings_10000BCE_2025_contact_sheet.png"

YODA_BASE = (
    "https://geo.public.data.uu.nl:443/vault-hyde/"
    "hyde35_c9_apr2025%5B1749214444%5D/original/gbc2025_7apr_base/"
)


@dataclass(frozen=True, slots=True)
class LandSignal:
    key: str
    label: str
    filename: str
    sector: tuple[float, float]
    color: tuple[int, int, int]


SIGNALS = [
    LandSignal("cropland", "Cropland", "his_crop_4apr2025.csv", (0.00, 0.33), (205, 149, 54)),
    LandSignal("pasture", "Pasture", "his_past_4apr2025.csv", (0.33, 0.66), (105, 126, 58)),
    LandSignal("rice", "Rice", "his_rice_4apr2025.csv", (0.66, 0.82), (91, 158, 140)),
    LandSignal("irrigation", "Irrigation", "his_irri_4apr2025.csv", (0.82, 1.00), (46, 111, 149)),
]


@dataclass(slots=True)
class RingRecord:
    index: int
    year: int
    totals: dict[str, float]
    footprint: float
    delta: float


@dataclass(slots=True)
class HydeModel:
    rings: list[RingRecord]
    boundaries: list[np.ndarray]
    cell_unit: float
    total_cells: int


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


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def closed(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points + points[:1]


def polar(radius: float, angle: float) -> tuple[float, float]:
    visual_angle = angle + ANGLE_OFFSET
    return (
        TREE_CENTER[0] + radius * math.cos(visual_angle),
        TREE_CENTER[1] + radius * math.sin(visual_angle),
    )


def download_signal(signal: LandSignal) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / signal.filename
    if not path.exists():
        request = urllib.request.Request(YODA_BASE + signal.filename, headers={"User-Agent": "curl/8.0"})
        path.write_bytes(urllib.request.urlopen(request, timeout=120).read())
    return path


def parse_year(column: str) -> int | None:
    if not column.startswith("y"):
        return None
    try:
        return int(column[1:])
    except ValueError:
        return None


def load_global_signals() -> pd.DataFrame:
    series_by_key: dict[str, pd.Series] = {}
    years: list[int] | None = None
    for signal in SIGNALS:
        df = pd.read_csv(download_signal(signal))
        year_columns = [column for column in df.columns if parse_year(column) is not None]
        parsed_years = [parse_year(column) for column in year_columns]
        keep = [(year, column) for year, column in zip(parsed_years, year_columns) if year is not None and year <= 2025]
        if years is None:
            years = [year for year, _column in keep]
        totals = df[[column for _year, column in keep]].sum(axis=0)
        totals.index = [year for year, _column in keep]
        series_by_key[signal.key] = totals.astype(float)

    if years is None:
        raise RuntimeError("No HYDE year columns found.")

    out = pd.DataFrame({"year": years})
    for signal in SIGNALS:
        out[signal.key] = out["year"].map(series_by_key[signal.key])
    out["primary_footprint"] = out["cropland"] + out["pasture"]
    out["delta_footprint"] = out["primary_footprint"].diff().fillna(0.0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    return out


def build_boundaries(rings: list[RingRecord]) -> list[np.ndarray]:
    footprints = np.array([max(1.0, ring.footprint) for ring in rings], dtype=float)
    widths = np.log1p(footprints)
    widths = widths / widths.sum() * (MAX_RADIUS - PITH_RADIUS)
    boundaries = [np.full(ANGLE_SAMPLES, PITH_RADIUS, dtype=float)]

    max_by_signal = {
        signal.key: max(max(ring.totals[signal.key] for ring in rings), 1.0)
        for signal in SIGNALS
    }

    for ring_index, ring in enumerate(rings):
        local = np.ones(ANGLE_SAMPLES, dtype=float) * 0.62
        for signal in SIGNALS:
            start, end = signal.sector
            strength = math.sqrt(max(0.0, ring.totals[signal.key]) / max_by_signal[signal.key])
            left = int(start * ANGLE_SAMPLES)
            right = int(end * ANGLE_SAMPLES)
            local[left:right] = 0.50 + 1.55 * strength
        local = circular_gaussian(local, sigma=12.0)
        local = np.clip(local, 0.38, 2.20)
        local /= local.mean()
        boundaries.append(boundaries[-1] + widths[ring_index] * local)
    return boundaries


def build_model() -> HydeModel:
    df = load_global_signals()
    rings = [
        RingRecord(
            index=index,
            year=int(row.year),
            totals={signal.key: float(getattr(row, signal.key)) for signal in SIGNALS},
            footprint=float(row.primary_footprint),
            delta=float(row.delta_footprint),
        )
        for index, row in enumerate(df.itertuples(index=False))
    ]
    max_signal = max(max(ring.totals[signal.key] for signal in SIGNALS) for ring in rings)
    cell_unit = max_signal / 2200.0
    total_cells = int(
        sum(max(0, round(ring.totals[signal.key] / cell_unit)) for ring in rings for signal in SIGNALS)
    )
    return HydeModel(rings=rings, boundaries=build_boundaries(rings), cell_unit=cell_unit, total_cells=total_cells)


def radius_at(model: HydeModel, ring_index: int, radial_t: np.ndarray, angle: np.ndarray) -> np.ndarray:
    inner = sample_periodic(model.boundaries[ring_index], angle)
    outer = sample_periodic(model.boundaries[ring_index + 1], angle)
    return inner + (outer - inner) * radial_t


def boundary_points(model: HydeModel, boundary_index: int, samples: int = 720) -> list[tuple[float, float]]:
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


def mix_with_white(color: tuple[int, int, int], amount: float) -> np.ndarray:
    base = np.array(color, dtype=float)
    white = np.array([255.0, 255.0, 255.0])
    return base * (1.0 - amount) + white * amount


def generate_frame_chunks(model: HydeModel) -> tuple[list[list[RectChunk]], np.ndarray]:
    chunks: list[list[RectChunk]] = [[] for _ in range(DRAW_FRAME_COUNT)]
    frame_counts = np.zeros(DRAW_FRAME_COUNT, dtype=np.int64)
    ring_count = len(model.rings)

    for ring in model.rings:
        for signal_index, signal in enumerate(SIGNALS):
            n = int(max(0, round(ring.totals[signal.key] / model.cell_unit)))
            if n <= 0:
                continue
            rng = np.random.default_rng(SEED + ring.index * 10_000 + signal_index * 103)
            start, end = signal.sector
            theta = rng.uniform(start * math.tau, end * math.tau, n)
            theta += rng.normal(0.0, math.tau * 0.010, n)
            radial_t = rng.random(n)
            theta += (radial_t - 0.5) * rng.normal(0.0, math.tau * 0.010, n)

            radius = radius_at(model, ring.index, radial_t, theta) + rng.normal(0.0, 0.08, n)
            visual_angle = theta + ANGLE_OFFSET
            x = np.rint(TREE_CENTER[0] + radius * np.cos(visual_angle) + rng.normal(0.0, 0.14, n)).astype(np.int32)
            y = np.rint(TREE_CENTER[1] + radius * np.sin(visual_angle) + rng.normal(0.0, 0.14, n)).astype(np.int32)

            age_t = ring.index / max(1, ring_count - 1)
            color = mix_with_white(signal.color, 0.36 - 0.22 * age_t)
            rgb = np.clip(color + rng.normal(0.0, 3.0, (n, 3)), 0, 255).astype(np.uint8)
            tx, ty = oriented_pixel_offsets(visual_angle)

            radial_progress = np.clip(radial_t + rng.normal(0.0, 0.012, n), 0.0, 0.998)
            progress = (ring.index + radial_progress) / ring_count
            birth_frames = np.clip((progress * (DRAW_FRAME_COUNT - 1)).astype(np.int32), 0, DRAW_FRAME_COUNT - 1)
            for frame_index in np.unique(birth_frames):
                mask = birth_frames == frame_index
                chunks[int(frame_index)].append(RectChunk(x=x[mask], y=y[mask], rgb=rgb[mask], tx=tx[mask], ty=ty[mask]))
                frame_counts[int(frame_index)] += int(mask.sum())
        if ring.index % 20 == 0:
            print(f"prepared HYDE ring {ring.index + 1}/{ring_count}: year {ring.year}")
    return chunks, frame_counts


def current_ring_index(model: HydeModel, frame_index: int) -> float:
    if frame_index >= DRAW_FRAME_COUNT:
        return float(len(model.rings))
    return min(len(model.rings), frame_index / max(1, DRAW_FRAME_COUNT - 1) * len(model.rings))


def draw_completed_boundaries(draw: ImageDraw.ImageDraw, model: HydeModel, current_index: float) -> None:
    completed = min(len(model.rings), int(math.floor(current_index)))
    for boundary_index in range(1, completed + 1):
        year = model.rings[boundary_index - 1].year
        if year in (0, 1000, 1500, 1700, 1800, 1900, 1950, 2000, 2025):
            alpha, width = 195, 2
        elif boundary_index == completed:
            alpha, width = 180, 2
        else:
            alpha, width = 92, 1
        draw.line(closed(boundary_points(model, boundary_index, samples=560)), fill=NEUTRAL_RING + (alpha,), width=width)


def draw_growth_front(draw: ImageDraw.ImageDraw, model: HydeModel, current_index: float) -> None:
    if current_index >= len(model.rings):
        draw.line(closed(boundary_points(model, len(model.rings), samples=760)), fill=NEUTRAL_RING + (232,), width=2)
        return
    ring_index = int(math.floor(current_index))
    frac = current_index - ring_index
    points = []
    for i in range(720):
        angle = i * math.tau / 720
        inner = float(sample_periodic(model.boundaries[ring_index], angle))
        outer = float(sample_periodic(model.boundaries[ring_index + 1], angle))
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


def render_all_frames(model: HydeModel, chunks: list[list[RectChunk]], frame_counts: np.ndarray) -> tuple[Image.Image, list[Image.Image]]:
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    final_frame: Image.Image | None = None
    contact_frames: list[Image.Image] = []
    contact_indices = {0, DRAW_FRAME_COUNT // 4, DRAW_FRAME_COUNT // 2, DRAW_FRAME_COUNT * 3 // 4, FRAME_COUNT - 1}
    generated_count = 0

    writer = cv2.VideoWriter(str(MP4_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer.")

    for frame_index in range(FRAME_COUNT):
        if frame_index < DRAW_FRAME_COUNT:
            for chunk in chunks[frame_index]:
                plot_rect_cells(canvas, chunk)
            generated_count += int(frame_counts[frame_index])

        frame = Image.fromarray(canvas).convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")
        idx_float = current_ring_index(model, frame_index)
        draw_pith(draw, min(frame_index, DRAW_FRAME_COUNT - 1))
        draw_completed_boundaries(draw, model, idx_float)
        draw_growth_front(draw, model, idx_float)

        rgb = frame.convert("RGB")
        writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        final_frame = rgb
        if frame_index in contact_indices:
            contact_frames.append(rgb.copy())
        if frame_index % max(1, FRAME_COUNT // 10) == 0:
            print(f"rendered {frame_index:04d}/{FRAME_COUNT} cells={generated_count:,}/{model.total_cells:,}")

    writer.release()
    if final_frame is None:
        raise RuntimeError("No frames rendered.")
    return final_frame, contact_frames


def save_contact_sheet(frames: list[Image.Image]) -> None:
    if not frames:
        return
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
        "18",
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
    model = build_model()
    print(f"HYDE rings: {model.rings[0].year} to {model.rings[-1].year} ({len(model.rings)} rings)")
    print(f"cell unit: {model.cell_unit:.2f} HYDE tabular units")
    print(f"total rendered cells: {model.total_cells:,}")
    chunks, frame_counts = generate_frame_chunks(model)
    final_frame, contact_frames = render_all_frames(model, chunks, frame_counts)
    final_frame.save(FINAL_FRAME_PATH)
    save_contact_sheet(contact_frames)
    encode_x_compatible()
    print(f"summary csv: {CSV_PATH}")
    print(f"video: {MP4_PATH} ({MP4_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"x video: {X_MP4_PATH} ({X_MP4_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"final frame: {FINAL_FRAME_PATH}")
    print(f"contact sheet: {CONTACT_SHEET_PATH}")
    for line in inspect_video(X_MP4_PATH):
        print(f"  {line}")


if __name__ == "__main__":
    main()
