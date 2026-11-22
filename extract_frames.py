from pathlib import Path

import cv2


def main():
    folder = Path(__file__).resolve().parent
    cap = cv2.VideoCapture(str(folder / "Video.mp4"))
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open Video.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 10:
            raise RuntimeError("Video must contain at least 10 frames")

        output = folder / "frames"
        output.mkdir(exist_ok=True)
        for i in range(10):
            frame_number = i * (total_frames - 1) // 9
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {frame_number}")
            if not cv2.imwrite(str(output / f"frame_{i + 1:02d}.png"), frame):
                raise RuntimeError("Cannot save frame")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
