from __future__ import annotations

import calendar
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import generate_belgium_weekly_mortality_tree as base


WIDTH = 1440
HEIGHT = 1080
FPS = 24
DRAW_SECONDS = 51
REVEAL_SECONDS = 9
DURATION_SECONDS = DRAW_SECONDS + REVEAL_SECONDS
DRAW_FRAME_COUNT = DRAW_SECONDS * FPS
FRAME_COUNT = DURATION_SECONDS * FPS

TREE_CENTER = (720, 620)
RADIUS_SCALE = 0.96
BACKGROUND = (255, 255, 255)
INK = (34, 34, 32)
MID = (96, 96, 90)
SOFT = (151, 145, 133)
NEUTRAL_RING = base.NEUTRAL_RING
WEEK_COUNT = base.WEEK_COUNT
MODEL = base.MODEL
SEED = 20240615

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MP4_PATH = OUTPUT_DIR / "belgium_mortality_rings_final_4x3_1992_2025.mp4"
X_MP4_PATH = OUTPUT_DIR / "belgium_mortality_rings_final_4x3_1992_2025_x_h264_aac.mp4"
FINAL_FRAME_PATH = OUTPUT_DIR / "belgium_mortality_rings_final_4x3_1992_2025_final_frame.png"
CONTACT_SHEET_PATH = OUTPUT_DIR / "belgium_mortality_rings_final_4x3_1992_2025_contact_sheet.png"


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    return base.load_font(size, bold=bold)


FONT_TITLE = load_font(52, bold=True)
FONT_SUBTITLE = load_font(24)
FONT_LABEL_DATE = load_font(18, bold=True)
FONT_LABEL_EVENT = load_font(16)
FONT_HANDLE = load_font(20, bold=True)


@dataclass(slots=True)
class RectChunk:
    x: np.ndarray
    y: np.ndarray
    rgb: np.ndarray
    tx: np.ndarray
    ty: np.ndarray


@dataclass(slots=True)
class EventLabel:
    title: str
    date_label: str
    year: int
    week_mid: float
    label_xy: tuple[int, int]
    align: str
    color: tuple[int, int, int]


EVENTS = [
    EventLabel("COVID-19 wave 1", "2020 Apr", 2020, 15.0, (1040, 778), "left", (74, 16, 34)),
    EventLabel("COVID-19 wave 2", "2020 Nov", 2020, 45.5, (510, 240), "right", (31, 6, 16)),
    EventLabel("August heatwave", "2020 Aug", 2020, 33.0, (390, 730), "right", (183, 101, 58)),
    EventLabel("Summer heat", "2022 Jul-Aug", 2022, 31.0, (520, 912), "center", (135, 43, 54)),
    EventLabel("Heatwave", "2003 Aug", 2003, 33.0, (438, 585), "right", (183, 101, 58)),
    EventLabel("A(H3N2) + cold", "2012 Mar", 2012, 8.0, (1038, 500), "left", (111, 66, 37)),
    EventLabel("Flu winter", "2015 Feb-Mar", 2015, 8.5, (1056, 410), "left", (47, 82, 102)),
    EventLabel("Flu B + cold", "2018 Mar", 2018, 10.0, (1064, 654), "left", (74, 16, 34)),
]


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def ease_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


def closed(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points + points[:1]


def polar(radius: float, angle: float) -> tuple[float, float]:
    visual_angle = angle + base.ANGLE_OFFSET
    return (
        TREE_CENTER[0] + radius * RADIUS_SCALE * math.cos(visual_angle),
        TREE_CENTER[1] + radius * RADIUS_SCALE * math.sin(visual_angle),
    )


def radius_at(year_index: int, radial_t: np.ndarray, angle: np.ndarray) -> np.ndarray:
    return base.radius_at(year_index, radial_t, angle) * RADIUS_SCALE


def boundary_points(boundary_index: int, samples: int = 720) -> list[tuple[float, float]]:
    values = MODEL.boundaries[boundary_index]
    phase = (boundary_index * 2.399963229728653) % math.tau
    return [
        polar(float(base.sample_periodic(values, phase + i * math.tau / samples)), phase + i * math.tau / samples)
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


def generate_frame_chunks() -> tuple[list[list[RectChunk]], np.ndarray]:
    chunks: list[list[RectChunk]] = [[] for _ in range(DRAW_FRAME_COUNT)]
    frame_counts = np.zeros(DRAW_FRAME_COUNT, dtype=np.int64)
    year_count = len(MODEL.years)

    for year_index, days in enumerate(MODEL.days_by_year):
        weekly_excess = np.zeros(WEEK_COUNT, dtype=float)
        for week in MODEL.weeks_by_year[year_index]:
            weekly_excess[week.week_index] = week.excess
        weekly_excess = base.circular_gaussian(weekly_excess, sigma=0.72)

        for day in days:
            n = day.deaths
            rng = np.random.default_rng(SEED + day.year * 10_000 + day.day_of_year)
            day_fraction = (day.day_of_year + 0.5) / day.days_in_year
            theta_center = math.tau * day_fraction
            theta = theta_center + rng.normal(0.0, math.tau * rng.uniform(0.28, 0.62) / day.days_in_year, n)
            radial_t = rng.random(n)
            theta += (radial_t - 0.5) * rng.normal(0.0, math.tau * 0.50 / day.days_in_year)

            radius = radius_at(year_index, radial_t, theta) + rng.normal(0.0, 0.08, n)
            visual_angle = theta + base.ANGLE_OFFSET
            x = np.rint(TREE_CENTER[0] + radius * np.cos(visual_angle) + rng.normal(0.0, 0.16, n)).astype(np.int32)
            y = np.rint(TREE_CENTER[1] + radius * np.sin(visual_angle) + rng.normal(0.0, 0.16, n)).astype(np.int32)

            week_float = (theta % math.tau) / math.tau * WEEK_COUNT
            excess = base.circular_interp(weekly_excess, week_float)
            base_color = base.colors_for_excess(excess)
            rgb = np.clip(base_color + rng.normal(0.0, 3.0, (n, 3)), 0, 255).astype(np.uint8)
            tx, ty = oriented_pixel_offsets(visual_angle)

            radial_progress = np.clip(radial_t + rng.normal(0.0, 0.014, n), 0.0, 0.998)
            progress = (year_index + radial_progress) / year_count
            birth_frames = np.clip((progress * (DRAW_FRAME_COUNT - 1)).astype(np.int32), 0, DRAW_FRAME_COUNT - 1)

            for frame_index in np.unique(birth_frames):
                mask = birth_frames == frame_index
                chunks[int(frame_index)].append(
                    RectChunk(
                        x=x[mask],
                        y=y[mask],
                        rgb=rgb[mask],
                        tx=tx[mask],
                        ty=ty[mask],
                    )
                )
                frame_counts[int(frame_index)] += int(mask.sum())

        if year_index % 5 == 0:
            print(f"prepared final cells through {MODEL.years[year_index]}")

    return chunks, frame_counts


def current_year_index(frame_index: int) -> float:
    if frame_index >= DRAW_FRAME_COUNT:
        return float(len(MODEL.years))
    return min(len(MODEL.years), frame_index / max(1, DRAW_FRAME_COUNT - 1) * len(MODEL.years))


def draw_completed_boundaries(draw: ImageDraw.ImageDraw, current_index: float) -> None:
    completed = min(len(MODEL.years), int(math.floor(current_index)))
    for boundary_index in range(1, completed + 1):
        year = MODEL.years[boundary_index - 1]
        if year in (2003, 2015, 2020, 2022):
            alpha, width = 225, 2
        elif year % 5 == 0 or boundary_index == completed:
            alpha, width = 188, 2
        else:
            alpha, width = 132, 1
        draw.line(closed(boundary_points(boundary_index, samples=560)), fill=NEUTRAL_RING + (alpha,), width=width)


def draw_growth_front(draw: ImageDraw.ImageDraw, current_index: float) -> None:
    if current_index >= len(MODEL.years):
        draw.line(closed(boundary_points(len(MODEL.years), samples=760)), fill=NEUTRAL_RING + (235,), width=2)
        return
    year_index = int(math.floor(current_index))
    frac = current_index - year_index
    points = []
    for i in range(720):
        angle = i * math.tau / 720
        inner = float(base.sample_periodic(MODEL.boundaries[year_index], angle))
        outer = float(base.sample_periodic(MODEL.boundaries[year_index + 1], angle))
        points.append(polar(inner + (outer - inner) * frac, angle))
    draw.line(closed(points), fill=NEUTRAL_RING + (226,), width=2)


def draw_pith(draw: ImageDraw.ImageDraw, frame_index: int) -> None:
    radius = base.PITH_RADIUS * RADIUS_SCALE * smoothstep(min(1.0, frame_index / 18.0))
    if radius <= 0:
        return
    box = (
        TREE_CENTER[0] - radius,
        TREE_CENTER[1] - radius,
        TREE_CENTER[0] + radius,
        TREE_CENTER[1] + radius,
    )
    draw.ellipse(box, fill=NEUTRAL_RING + (255,), outline=(112, 105, 96, 145), width=1)


def event_anchor(event: EventLabel) -> tuple[float, float]:
    year_index = MODEL.years.index(event.year)
    days_in_year = 366 if calendar.isleap(event.year) else 365
    theta = math.tau * ((event.week_mid - 0.5) / WEEK_COUNT)
    if event.week_mid > WEEK_COUNT:
        theta = math.tau * ((event.week_mid - 0.5) / (days_in_year / 7.0))
    radius = float(radius_at(year_index, np.array([0.68]), np.array([theta]))[0])
    visual_angle = theta + base.ANGLE_OFFSET
    return (
        TREE_CENTER[0] + radius * math.cos(visual_angle),
        TREE_CENTER[1] + radius * math.sin(visual_angle),
    )


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, font: ImageFont.ImageFont, fill: tuple[int, int, int, int]) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((WIDTH - (box[2] - box[0])) / 2, y), text, font=font, fill=fill)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_label(
    draw: ImageDraw.ImageDraw,
    event: EventLabel,
    progress: float,
    *,
    draw_leader: bool = True,
    draw_box: bool = True,
) -> None:
    progress = ease_out_cubic(progress)
    if progress <= 0:
        return
    alpha = int(255 * progress)
    anchor = event_anchor(event)

    date_w, date_h = text_size(draw, event.date_label, FONT_LABEL_DATE)
    event_w, event_h = text_size(draw, event.title, FONT_LABEL_EVENT)
    box_w = max(date_w, event_w) + 36
    box_h = 62
    label_x, label_y = event.label_xy
    label_y += int((1.0 - progress) * 7)
    if event.align == "right":
        box_x = label_x - box_w
        line_end = (box_x + box_w, label_y + box_h / 2)
    elif event.align == "center":
        box_x = label_x - box_w // 2
        line_end = (box_x + box_w / 2, label_y)
    else:
        box_x = label_x
        line_end = (box_x, label_y + box_h / 2)

    partial_end = (
        anchor[0] + (line_end[0] - anchor[0]) * progress,
        anchor[1] + (line_end[1] - anchor[1]) * progress,
    )
    if draw_leader:
        draw.line((anchor[0], anchor[1], partial_end[0], partial_end[1]), fill=event.color + (int(178 * progress),), width=2)
        r = 4 + 3 * progress
        draw.ellipse((anchor[0] - r, anchor[1] - r, anchor[0] + r, anchor[1] + r), fill=event.color + (alpha,), outline=(255, 255, 255, alpha), width=2)

    if not draw_box or progress < 0.35:
        return
    text_alpha = int(255 * smoothstep((progress - 0.35) / 0.65))
    bg_alpha = int(232 * smoothstep((progress - 0.25) / 0.75))
    box = (box_x, label_y, box_x + box_w, label_y + box_h)
    draw.rounded_rectangle(box, radius=7, fill=(255, 255, 255, bg_alpha), outline=event.color + (int(126 * progress),), width=1)
    draw.ellipse((box_x + 13, label_y + 17, box_x + 21, label_y + 25), fill=event.color + (text_alpha,))
    draw.text((box_x + 30, label_y + 10), event.date_label, font=FONT_LABEL_DATE, fill=INK + (text_alpha,))
    draw.text((box_x + 30, label_y + 36), event.title, font=FONT_LABEL_EVENT, fill=MID + (text_alpha,))


def draw_centered_with_shadow(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    x = (WIDTH - (box[2] - box[0])) / 2
    shadow_alpha = max(0, min(72, fill[3] // 5))
    draw.text((x, y + 2), text, font=font, fill=(190, 178, 150, shadow_alpha))
    draw.text((x, y), text, font=font, fill=fill)


def draw_final_overlay(frame: Image.Image, frame_index: int) -> None:
    if frame_index < DRAW_FRAME_COUNT:
        return
    elapsed = (frame_index - DRAW_FRAME_COUNT) / FPS
    draw = ImageDraw.Draw(frame, "RGBA")

    title_progress = smoothstep(elapsed / 1.9)
    subtitle_progress = smoothstep((elapsed - 0.45) / 1.7)
    title_alpha = int(255 * title_progress)
    subtitle_alpha = int(255 * subtitle_progress)
    title_y = 24 + int((1.0 - title_progress) * 18)
    subtitle_y = 89 + int((1.0 - subtitle_progress) * 8)
    draw_centered_with_shadow(draw, "Belgium Mortality Rings", title_y, FONT_TITLE, INK + (title_alpha,))
    draw_centered(
        draw,
        "A dendrochronology-inspired view of weekly deaths, 1992-2025",
        subtitle_y,
        FONT_SUBTITLE,
        MID + (subtitle_alpha,),
    )

    accent_progress = smoothstep((elapsed - 0.95) / 1.3)
    if accent_progress > 0:
        half = int(230 * accent_progress)
        y = 133
        cx = WIDTH // 2
        draw.line((cx - half, y, cx - 18, y), fill=(214, 168, 79, int(190 * accent_progress)), width=2)
        draw.line((cx + 18, y, cx + half, y), fill=(135, 43, 54, int(190 * accent_progress)), width=2)
        draw.ellipse((cx - 4, y - 4, cx + 4, y + 4), fill=NEUTRAL_RING + (int(210 * accent_progress),))

    label_states = [(event, (elapsed - 1.75 - index * 0.42) / 0.82) for index, event in enumerate(EVENTS)]
    for event, label_progress in label_states:
        draw_label(draw, event, label_progress, draw_box=False)
    for event, label_progress in label_states:
        draw_label(draw, event, label_progress, draw_leader=False)

    handle_alpha = int(255 * smoothstep((elapsed - 6.4) / 1.2))
    handle = "@FlorentChif"
    handle_box = draw.textbbox((0, 0), handle, font=FONT_HANDLE)
    draw.text((WIDTH - (handle_box[2] - handle_box[0]) - 40, HEIGHT - 48), handle, font=FONT_HANDLE, fill=(27, 38, 59, handle_alpha))


def render_all_frames(chunks: list[list[RectChunk]], frame_counts: np.ndarray) -> tuple[Image.Image, list[Image.Image]]:
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    final_frame: Image.Image | None = None
    contact_frames: list[Image.Image] = []
    contact_indices = {0, DRAW_FRAME_COUNT // 2, DRAW_FRAME_COUNT - 1, DRAW_FRAME_COUNT + FPS * 4, FRAME_COUNT - 1}
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
        idx_float = current_year_index(frame_index)
        draw_pith(draw, min(frame_index, DRAW_FRAME_COUNT - 1))
        draw_completed_boundaries(draw, idx_float)
        draw_growth_front(draw, idx_float)
        draw_final_overlay(frame, frame_index)

        rgb = frame.convert("RGB")
        writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        final_frame = rgb
        if frame_index in contact_indices:
            contact_frames.append(rgb.copy())
        if frame_index % max(1, FRAME_COUNT // 12) == 0:
            print(f"rendered {frame_index:04d}/{FRAME_COUNT} cells={generated_count:,}/{MODEL.total_deaths:,}")

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
    labels = [f"{seconds:g}s" for seconds in [0, DRAW_SECONDS / 2, DRAW_SECONDS, DRAW_SECONDS + 4, DURATION_SECONDS]]
    for index, frame in enumerate(frames):
        x = padding + index * (thumb_w + padding)
        y = padding
        sheet.paste(frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + thumb_h + 6), labels[index], font=base.FONT_SMALL, fill=MID)
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
    print(f"years: {MODEL.years[0]}-{MODEL.years[-1]} ({len(MODEL.years)} rings)")
    print(f"total cells: {MODEL.total_deaths:,}")
    print(f"format: {WIDTH}x{HEIGHT} 4:3, {DURATION_SECONDS}s total, {DRAW_SECONDS}s draw")
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
