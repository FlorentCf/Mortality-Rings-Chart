from __future__ import annotations

from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
SOURCE = OUTPUT_DIR / "tree_growth_chronology_30s.mp4"

WEB_EXPORTS = [
    {
        "path": OUTPUT_DIR / "tree_growth_chronology_web_480p_15fps.mp4",
        "size": (854, 480),
        "fps": 15,
    },
    {
        "path": OUTPUT_DIR / "tree_growth_chronology_web_360p_12fps.mp4",
        "size": (640, 360),
        "fps": 12,
    },
]


def export_version(source: Path, target: Path, size: tuple[int, int], target_fps: int) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = source_frames / source_fps
    target_frames = round(duration * target_fps)

    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {target}")

    for frame_index in range(target_frames):
        source_index = min(source_frames - 1, round(frame_index * source_fps / target_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {source_index} from {source}")
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(resized)

    cap.release()
    writer.release()


def describe(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frames / fps if fps else 0
    size_mb = path.stat().st_size / 1024 / 1024
    return f"{path.name}: {size_mb:.2f} MB, {width}x{height}, {fps:.0f} fps, {duration:.1f}s"


def main() -> None:
    for export in WEB_EXPORTS:
        export_version(SOURCE, export["path"], export["size"], export["fps"])
        print(describe(export["path"]))


if __name__ == "__main__":
    main()
