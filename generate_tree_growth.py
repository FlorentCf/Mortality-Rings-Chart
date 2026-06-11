from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 24
DURATION_SECONDS = 30
FRAME_COUNT = FPS * DURATION_SECONDS

SEED = 42
YEARS = 15
ANGLE_SAMPLES = 960

TREE_CENTER = (405, 360)
PITH_RADIUS = 18.0
TARGET_MEAN_RADIUS = 300.0

BACKGROUND = (255, 255, 255)
INK = (38, 38, 38)
MID = (105, 105, 105)
LIGHT = (190, 190, 190)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MP4_PATH = OUTPUT_DIR / "tree_growth_chronology_30s.mp4"
FINAL_FRAME_PATH = OUTPUT_DIR / "tree_growth_chronology_final_frame.png"
CONTACT_SHEET_PATH = OUTPUT_DIR / "tree_growth_chronology_contact_sheet.png"


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


FONT_TITLE = load_font(30, bold=True)
FONT_SECTION = load_font(22, bold=True)
FONT_BODY = load_font(17)
FONT_SMALL = load_font(13)
FONT_TINY = load_font(11)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def polar(radius: float, angle: float, center: tuple[int, int] = TREE_CENTER) -> tuple[float, float]:
    return (
        center[0] + radius * math.cos(angle),
        center[1] + radius * math.sin(angle),
    )


def closed_polyline(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return points
    return points + [points[0]]


def periodic_noise(
    rng: np.random.Generator,
    theta: np.ndarray,
    harmonics: tuple[int, ...],
    falloff: float = 1.0,
) -> np.ndarray:
    values = np.zeros_like(theta)
    for harmonic in harmonics:
        phase = rng.uniform(0.0, math.tau)
        amplitude = rng.normal(0.0, 1.0) / (harmonic**falloff)
        values += amplitude * np.sin(theta * harmonic + phase)
    values -= values.mean()
    std = values.std()
    if std > 1e-9:
        values /= std
    return values


@dataclass(slots=True)
class GrowthModel:
    theta: np.ndarray
    boundaries: list[np.ndarray]
    latewood_starts: list[np.ndarray]
    annual_widths: list[np.ndarray]


@dataclass(slots=True)
class Cell:
    year: int
    theta0: float
    theta1: float
    t0: float
    t1: float
    birth_frame: int
    mature_frame: int
    fill: tuple[int, int, int, int]
    outline: tuple[int, int, int, int]
    active_outline: tuple[int, int, int, int]
    width: int
    radial_wobble: float
    theta_wobble: float
    latewood: bool


def build_growth_model() -> GrowthModel:
    rng = np.random.default_rng(SEED)
    theta = np.linspace(0.0, math.tau, ANGLE_SAMPLES, endpoint=False)

    pith_lobing = 1.7 * periodic_noise(rng, theta, (2, 3, 5), falloff=0.7)
    boundaries = [np.full_like(theta, PITH_RADIUS) + pith_lobing]

    base_widths = np.linspace(23.5, 17.5, YEARS)
    base_widths *= rng.lognormal(mean=0.0, sigma=0.13, size=YEARS)
    base_widths *= (TARGET_MEAN_RADIUS - PITH_RADIUS) / base_widths.sum()

    latewood_starts: list[np.ndarray] = []
    annual_widths: list[np.ndarray] = []
    persistent_lean = 0.42 * periodic_noise(rng, theta, (1, 2), falloff=0.2)

    for year in range(YEARS):
        local_weather = periodic_noise(rng, theta, (1, 2, 3, 5, 8, 13), falloff=0.8)
        sector = np.sin(theta - rng.uniform(0.0, math.tau))
        sector_stress = np.maximum(0.0, sector) ** 2
        amplitude = 0.16 + 0.05 * math.sin(year * 1.3)
        width = base_widths[year] * (
            1.0
            + amplitude * local_weather
            + 0.10 * persistent_lean
            - 0.13 * sector_stress * (0.5 + 0.5 * math.sin(year * 0.9))
        )
        width = np.clip(width, base_widths[year] * 0.52, base_widths[year] * 1.65)
        annual_widths.append(width)

        late_fraction = 0.68 + 0.055 * periodic_noise(rng, theta, (2, 3, 6), falloff=0.6)
        late_fraction = np.clip(late_fraction, 0.58, 0.78)
        inner = boundaries[-1]
        latewood_starts.append(inner + width * late_fraction)
        boundaries.append(inner + width)

    return GrowthModel(
        theta=theta,
        boundaries=boundaries,
        latewood_starts=latewood_starts,
        annual_widths=annual_widths,
    )


MODEL = build_growth_model()


def sample_periodic(values: np.ndarray, angle: float) -> float:
    position = (angle % math.tau) / math.tau * len(values)
    left = int(math.floor(position)) % len(values)
    right = (left + 1) % len(values)
    frac = position - math.floor(position)
    return float(values[left] * (1.0 - frac) + values[right] * frac)


def radius_at(year: int, radial_t: float, angle: float) -> float:
    if year < 0:
        return sample_periodic(MODEL.boundaries[0], angle) * radial_t
    inner = sample_periodic(MODEL.boundaries[year], angle)
    outer = sample_periodic(MODEL.boundaries[year + 1], angle)
    return inner + (outer - inner) * radial_t


def ring_boundary_points(year: int, outer: bool = True, count: int = 360) -> list[tuple[float, float]]:
    values = MODEL.boundaries[year + 1 if outer else year]
    points = []
    for index in range(count):
        angle = index * math.tau / count
        points.append(polar(sample_periodic(values, angle), angle))
    return points


def make_cell_polygon(cell: Cell, growth: float = 1.0) -> list[tuple[float, float]]:
    growth = smoothstep(growth)
    current_t1 = cell.t0 + (cell.t1 - cell.t0) * growth
    theta0 = cell.theta0 + cell.theta_wobble * 0.35
    theta1 = cell.theta1 + cell.theta_wobble * -0.35
    theta_mid = (theta0 + theta1) * 0.5

    if theta1 < theta0:
        theta1 += math.tau
        theta_mid = (theta0 + theta1) * 0.5

    t_mid = (cell.t0 + current_t1) * 0.5
    wobble = cell.radial_wobble * math.sin(theta_mid * 3.0 + cell.year)

    def point(t_value: float, angle: float, offset: float = 0.0) -> tuple[float, float]:
        return polar(radius_at(cell.year, t_value, angle) + offset, angle)

    return [
        point(cell.t0, theta0, -0.20 * wobble),
        point(cell.t0, theta_mid, -0.12 * wobble),
        point(cell.t0, theta1, 0.10 * wobble),
        point(t_mid, theta1, 0.18 * wobble),
        point(current_t1, theta1, 0.28 * wobble),
        point(current_t1, theta_mid, 0.15 * wobble),
        point(current_t1, theta0, -0.20 * wobble),
        point(t_mid, theta0, -0.24 * wobble),
    ]


def draw_cell(draw: ImageDraw.ImageDraw, cell: Cell, growth: float = 1.0, active: bool = False) -> None:
    polygon = make_cell_polygon(cell, growth)
    if len(polygon) < 3:
        return

    fill = cell.fill
    outline = cell.active_outline if active else cell.outline
    if active:
        age_alpha = int(70 + 160 * smoothstep(growth))
        fill = (fill[0], fill[1], fill[2], max(fill[3], age_alpha))
    draw.polygon(polygon, fill=fill, outline=outline)


def generate_cells() -> list[Cell]:
    rng = np.random.default_rng(SEED + 1000)
    cells: list[Cell] = []

    # The pith starts as a cluster of irregular cells rather than a perfect disk.
    for ring_index, (inner_t, outer_t, count) in enumerate(
        [(0.0, 0.34, 10), (0.34, 0.68, 16), (0.68, 1.0, 22)]
    ):
        widths = rng.lognormal(mean=0.0, sigma=0.25, size=count)
        edges = np.concatenate(([0.0], np.cumsum(widths / widths.sum() * math.tau)))
        phase = rng.uniform(0.0, math.tau)
        for index in range(count):
            theta0 = float(edges[index] + phase)
            theta1 = float(edges[index + 1] + phase)
            birth = int(rng.integers(0, 24 + ring_index * 18))
            cells.append(
                Cell(
                    year=-1,
                    theta0=theta0,
                    theta1=theta1,
                    t0=max(0.0, inner_t - 0.018),
                    t1=min(1.0, outer_t + 0.018),
                    birth_frame=birth,
                    mature_frame=birth + int(rng.integers(10, 22)),
                    fill=(240, 240, 240, 255),
                    outline=(145, 145, 145, 120),
                    active_outline=(70, 70, 70, 210),
                    width=1,
                    radial_wobble=float(rng.normal(0.0, 0.6)),
                    theta_wobble=float(rng.normal(0.0, 0.003)),
                    latewood=False,
                )
            )

    for year in range(YEARS):
        width_mean = float(MODEL.annual_widths[year].mean())
        late_fraction_mean = float(
            ((MODEL.latewood_starts[year] - MODEL.boundaries[year]) / MODEL.annual_widths[year]).mean()
        )

        row_edges = [0.0]
        t = 0.0
        while t < 1.0 - 1e-6:
            is_late = t > late_fraction_mean
            target_height = rng.uniform(5.0, 7.4) if not is_late else rng.uniform(2.2, 3.6)
            target_height *= rng.uniform(0.85, 1.22)
            t = min(1.0, t + target_height / max(8.0, width_mean))
            row_edges.append(t)

        for row_index, (t0, t1) in enumerate(zip(row_edges[:-1], row_edges[1:])):
            t_mid = (t0 + t1) * 0.5
            latewood = t_mid >= late_fraction_mean
            avg_radius = float((MODEL.boundaries[year] + MODEL.annual_widths[year] * t_mid).mean())
            cell_width = rng.uniform(9.2, 12.2) if not latewood else rng.uniform(6.8, 9.2)
            angular_count = max(12, int((math.tau * avg_radius / cell_width) * rng.uniform(0.88, 1.11)))

            angular_widths = rng.lognormal(mean=0.0, sigma=0.23 if not latewood else 0.18, size=angular_count)
            angular_widths = angular_widths / angular_widths.sum() * math.tau
            theta_edges = np.concatenate(([0.0], np.cumsum(angular_widths)))
            theta_edges += rng.uniform(0.0, math.tau / angular_count)

            for column in range(angular_count):
                theta0 = float(theta_edges[column])
                theta1 = float(theta_edges[column + 1])
                theta_mid = (theta0 + theta1) * 0.5
                local_late_fraction = (
                    sample_periodic(MODEL.latewood_starts[year], theta_mid)
                    - sample_periodic(MODEL.boundaries[year], theta_mid)
                ) / max(1e-6, sample_periodic(MODEL.annual_widths[year], theta_mid))
                local_latewood = t_mid >= local_late_fraction

                local_surge = 0.030 * math.sin(theta_mid * 2.0 + year * 1.4)
                row_jitter = rng.normal(0.0, 0.018)
                local_timing = max(0.0, min(0.995, t0 + local_surge + row_jitter))
                birth_frame = int(((year + local_timing) / YEARS) * (FRAME_COUNT - 18))
                mature_delay = int(rng.integers(8, 19 if not local_latewood else 14))

                if local_latewood:
                    base = int(rng.integers(217, 232))
                    outline = int(rng.integers(118, 146))
                    fill = (base, base, base, 255)
                    line = (outline, outline, outline, 150)
                    active_line = (54, 54, 54, 225)
                else:
                    base = int(rng.integers(239, 250))
                    outline = int(rng.integers(155, 185))
                    fill = (base, base, base, 255)
                    line = (outline, outline, outline, 105)
                    active_line = (72, 72, 72, 205)

                cells.append(
                    Cell(
                        year=year,
                        theta0=theta0,
                        theta1=theta1,
                        t0=t0,
                        t1=t1,
                        birth_frame=birth_frame,
                        mature_frame=min(FRAME_COUNT - 1, birth_frame + mature_delay),
                        fill=fill,
                        outline=line,
                        active_outline=active_line,
                        width=1,
                        radial_wobble=float(rng.normal(0.0, 0.42 if not local_latewood else 0.26)),
                        theta_wobble=float(rng.normal(0.0, 0.004)),
                        latewood=local_latewood,
                    )
                )

    cells.sort(key=lambda cell: (cell.birth_frame, cell.mature_frame))
    return cells


CELLS = generate_cells()
TOTAL_CELLS = len(CELLS)


def current_year_at_frame(frame_index: int) -> float:
    return min(YEARS, frame_index / max(1, FRAME_COUNT - 1) * YEARS)


def draw_completed_boundaries(draw: ImageDraw.ImageDraw, current_year: float) -> None:
    completed = int(min(YEARS, math.floor(current_year)))
    for year in range(1, completed + 1):
        points = ring_boundary_points(year - 1, outer=True, count=420)
        shade = 115 if year % 5 else 70
        alpha = 150 if year % 5 else 190
        width = 1 if year % 5 else 2
        draw.line(closed_polyline(points), fill=(shade, shade, shade, alpha), width=width, joint="curve")


def draw_growth_front(draw: ImageDraw.ImageDraw, current_year: float) -> None:
    year = min(YEARS - 1, max(0, int(math.floor(current_year))))
    frac = current_year - year
    if current_year >= YEARS:
        year = YEARS - 1
        frac = 1.0

    points = []
    for index in range(320):
        angle = index * math.tau / 320
        radius = radius_at(year, frac, angle)
        points.append(polar(radius, angle))
    draw.line(closed_polyline(points), fill=(55, 55, 55, 170), width=2, joint="curve")


def draw_annotations(
    draw: ImageDraw.ImageDraw,
    frame_index: int,
    generated_count: int,
    active_count: int,
) -> None:
    current_year = current_year_at_frame(frame_index)
    current_year_clamped = min(YEARS, current_year)
    year_index = min(YEARS, max(1, math.ceil(current_year_clamped)))
    phase_t = current_year_clamped - math.floor(current_year_clamped)
    phase = "latewood band" if phase_t > 0.70 and current_year_clamped < YEARS else "earlywood cells"
    if current_year_clamped >= YEARS:
        phase = "outer cambium edge"

    panel_x = 790
    draw.text((panel_x, 72), "Tree growth chronology", font=FONT_TITLE, fill=INK)
    draw.text((panel_x, 109), "individual cell births with irregular annual rings", font=FONT_BODY, fill=MID)

    draw.text((panel_x, 156), f"year {year_index:02d} / {YEARS}", font=FONT_SECTION, fill=INK)
    draw.text((panel_x, 184), phase, font=FONT_BODY, fill=MID)

    stat_y = 230
    draw.text((panel_x, stat_y), "cells generated", font=FONT_SMALL, fill=MID)
    draw.text((panel_x + 140, stat_y - 4), f"{generated_count:,} / {TOTAL_CELLS:,}", font=FONT_SECTION, fill=INK)
    draw.text((panel_x, stat_y + 34), "currently growing", font=FONT_SMALL, fill=MID)
    draw.text((panel_x + 140, stat_y + 30), f"{active_count:,}", font=FONT_SECTION, fill=INK)

    rail_x = panel_x + 32
    rail_top = 320
    rail_bottom = 615
    rail_height = rail_bottom - rail_top
    draw.line((rail_x, rail_top, rail_x, rail_bottom), fill=(170, 170, 170, 180), width=2)

    for year in range(YEARS + 1):
        y = rail_top + rail_height * year / YEARS
        completed = current_year_clamped >= year
        radius = 5 if year % 5 else 7
        fill = (58, 58, 58, 235) if completed else (236, 236, 236, 255)
        outline = (86, 86, 86, 220) if completed else (182, 182, 182, 180)
        draw.ellipse((rail_x - radius, y - radius, rail_x + radius, y + radius), fill=fill, outline=outline, width=1)
        if year == 0:
            label = "pith"
        elif year % 3 == 0 or year == YEARS:
            label = f"year {year}"
        else:
            label = ""
        if label:
            draw.text((rail_x + 18, y - 9), label, font=FONT_SMALL, fill=MID)

    marker_y = rail_top + rail_height * current_year_clamped / YEARS
    draw.rounded_rectangle((rail_x - 18, marker_y - 10, rail_x + 18, marker_y + 10), radius=10, fill=(54, 54, 54, 235))
    label_y = marker_y - 31 if current_year_clamped > YEARS - 0.8 else marker_y - 9
    draw.text((rail_x + 50, label_y), "growth front", font=FONT_SMALL, fill=INK)

    draw.line((72, 664, 694, 664), fill=(214, 214, 214, 200), width=1)
    draw.text(
        (72, 680),
        "Each visible cell is added by birth frame; annual bands vary by angle and local cell size.",
        font=FONT_SMALL,
        fill=(118, 118, 118),
    )


def render_all_frames() -> tuple[Image.Image, list[Image.Image]]:
    permanent_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    birth_cursor = 0
    active_cells: list[Cell] = []
    final_frame: Image.Image | None = None
    contact_frames: list[Image.Image] = []
    contact_indices = {0, FRAME_COUNT // 4, FRAME_COUNT // 2, FRAME_COUNT * 3 // 4, FRAME_COUNT - 1}

    writer = cv2.VideoWriter(
        str(MP4_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer with the mp4v codec.")

    for frame_index in range(FRAME_COUNT):
        while birth_cursor < TOTAL_CELLS and CELLS[birth_cursor].birth_frame <= frame_index:
            active_cells.append(CELLS[birth_cursor])
            birth_cursor += 1

        permanent_draw = ImageDraw.Draw(permanent_layer, "RGBA")
        still_growing: list[Cell] = []
        for cell in active_cells:
            if cell.mature_frame <= frame_index:
                draw_cell(permanent_draw, cell, growth=1.0, active=False)
            else:
                still_growing.append(cell)
        active_cells = still_growing

        frame = Image.new("RGBA", (WIDTH, HEIGHT), BACKGROUND + (255,))
        frame.alpha_composite(permanent_layer)
        draw = ImageDraw.Draw(frame, "RGBA")

        for cell in active_cells:
            growth = (frame_index - cell.birth_frame) / max(1, cell.mature_frame - cell.birth_frame)
            draw_cell(draw, cell, growth=growth, active=True)

        current_year = current_year_at_frame(frame_index)
        draw_completed_boundaries(draw, current_year)
        draw_growth_front(draw, current_year)
        draw_annotations(draw, frame_index, generated_count=birth_cursor, active_count=len(active_cells))

        rgb = frame.convert("RGB")
        writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        final_frame = rgb

        if frame_index in contact_indices:
            contact_frames.append(rgb.copy())

        if frame_index % max(1, FRAME_COUNT // 10) == 0:
            print(
                f"rendered {frame_index:04d}/{FRAME_COUNT} "
                f"cells={birth_cursor}/{TOTAL_CELLS} active={len(active_cells)}"
            )

    writer.release()

    if final_frame is None:
        raise RuntimeError("No frames were rendered.")

    return final_frame, contact_frames


def save_contact_sheet(frames: list[Image.Image]) -> None:
    if not frames:
        return

    thumb_w = WIDTH // 3
    thumb_h = HEIGHT // 3
    padding = 18
    label_h = 22
    sheet_w = thumb_w * len(frames) + padding * (len(frames) + 1)
    sheet_h = thumb_h + padding * 2 + label_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    labels = ["0s", "7.5s", "15s", "22.5s", "30s"]

    for index, frame in enumerate(frames):
        x = padding + index * (thumb_w + padding)
        y = padding
        thumb = frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        draw.text((x, y + thumb_h + 5), labels[index] if index < len(labels) else f"{index}", font=FONT_SMALL, fill=MID)

    sheet.save(CONTACT_SHEET_PATH)


def write_video() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"generated cell events: {TOTAL_CELLS:,}")
    final_frame, contact_frames = render_all_frames()
    final_frame.save(FINAL_FRAME_PATH)
    save_contact_sheet(contact_frames)

    print(f"rendered {FRAME_COUNT} frames at {FPS} fps")
    print(f"video: {MP4_PATH}")
    print(f"final frame: {FINAL_FRAME_PATH}")
    print(f"contact sheet: {CONTACT_SHEET_PATH}")


if __name__ == "__main__":
    write_video()
