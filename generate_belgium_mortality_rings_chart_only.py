from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

import generate_belgium_mortality_rings_final as final


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MP4_PATH = OUTPUT_DIR / "belgium_mortality_rings_chart_only_4x3_1992_2025.mp4"
X_MP4_PATH = OUTPUT_DIR / "belgium_mortality_rings_chart_only_4x3_1992_2025_x_h264_aac.mp4"
FINAL_FRAME_PATH = OUTPUT_DIR / "belgium_mortality_rings_chart_only_4x3_1992_2025_final_frame.png"
CONTACT_SHEET_PATH = OUTPUT_DIR / "belgium_mortality_rings_chart_only_4x3_1992_2025_contact_sheet.png"


def render_all_frames(
    chunks: list[list[final.RectChunk]],
    frame_counts: np.ndarray,
) -> tuple[Image.Image, list[Image.Image]]:
    canvas = np.full((final.HEIGHT, final.WIDTH, 3), 255, dtype=np.uint8)
    final_frame: Image.Image | None = None
    contact_frames: list[Image.Image] = []
    contact_indices = {
        0,
        final.DRAW_FRAME_COUNT // 4,
        final.DRAW_FRAME_COUNT // 2,
        final.DRAW_FRAME_COUNT * 3 // 4,
        final.FRAME_COUNT - 1,
    }
    generated_count = 0

    writer = cv2.VideoWriter(str(MP4_PATH), cv2.VideoWriter_fourcc(*"mp4v"), final.FPS, (final.WIDTH, final.HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer.")

    for frame_index in range(final.FRAME_COUNT):
        if frame_index < final.DRAW_FRAME_COUNT:
            for chunk in chunks[frame_index]:
                final.plot_rect_cells(canvas, chunk)
            generated_count += int(frame_counts[frame_index])

        frame = Image.fromarray(canvas).convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")
        idx_float = final.current_year_index(frame_index)
        final.draw_pith(draw, min(frame_index, final.DRAW_FRAME_COUNT - 1))
        final.draw_completed_boundaries(draw, idx_float)
        final.draw_growth_front(draw, idx_float)

        rgb = frame.convert("RGB")
        writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        final_frame = rgb
        if frame_index in contact_indices:
            contact_frames.append(rgb.copy())
        if frame_index % max(1, final.FRAME_COUNT // 12) == 0:
            print(f"rendered {frame_index:04d}/{final.FRAME_COUNT} cells={generated_count:,}/{final.MODEL.total_deaths:,}")

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
        final.BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    labels = [f"{seconds:g}s" for seconds in [0, final.DRAW_SECONDS / 4, final.DRAW_SECONDS / 2, final.DRAW_SECONDS * 3 / 4, final.DURATION_SECONDS]]
    for index, frame in enumerate(frames):
        x = padding + index * (thumb_w + padding)
        y = padding
        sheet.paste(frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + thumb_h + 6), labels[index], font=final.base.FONT_SMALL, fill=final.MID)
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
    print(f"years: {final.MODEL.years[0]}-{final.MODEL.years[-1]} ({len(final.MODEL.years)} rings)")
    print(f"total cells: {final.MODEL.total_deaths:,}")
    print(f"chart only: {final.WIDTH}x{final.HEIGHT} 4:3, no title, no labels")
    chunks, frame_counts = final.generate_frame_chunks()
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
