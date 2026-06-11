from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

EXPORTS = [
    (
        OUTPUT_DIR / "tree_growth_chronology_30s.mp4",
        OUTPUT_DIR / "tree_growth_chronology_x_720p_h264_aac.mp4",
    ),
    (
        OUTPUT_DIR / "tree_growth_chronology_web_480p_15fps.mp4",
        OUTPUT_DIR / "tree_growth_chronology_x_480p_h264_aac.mp4",
    ),
    (
        OUTPUT_DIR / "tree_growth_chronology_web_360p_12fps.mp4",
        OUTPUT_DIR / "tree_growth_chronology_x_360p_h264_aac.mp4",
    ),
]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=True)


def encode_for_x(source: Path, target: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
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
        str(target),
    ]
    run(command)


def inspect_streams(path: Path) -> list[str]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    lines = []
    for line in completed.stderr.splitlines():
        clean = line.strip()
        if "Video:" in clean or "Audio:" in clean or "Duration:" in clean:
            lines.append(clean)
    return lines


def main() -> None:
    for source, target in EXPORTS:
        encode_for_x(source, target)
        print(f"{target.name}: {target.stat().st_size / 1024 / 1024:.2f} MB")
        for line in inspect_streams(target):
            print(f"  {line}")


if __name__ == "__main__":
    main()
