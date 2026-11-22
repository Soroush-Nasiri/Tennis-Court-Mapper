import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


FOLDER = Path(__file__).resolve().parent


def get_player_point(frame, model):
    """Return the bottom-center pixel and box of the strongest person detection."""
    result = model.predict(
        frame, imgsz=1280, classes=[0], conf=0.3, max_det=1, verbose=False
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None, None

    box = result.boxes.xyxy[0].cpu().numpy()
    if not np.isfinite(box).all():
        return None, None
    x1, _, x2, y2 = box
    point = (float((x1 + x2) / 2), float(y2))
    return point, np.rint(box).astype(int)


def pixel_to_world(point, H):
    """Map a pixel to ground coordinates in meters, with Y capped at 24."""
    ground = H @ np.array([point[0], point[1], 1.0])
    if not np.isfinite(ground).all() or abs(ground[2]) < 1e-8:
        return None
    X, Y = ground[:2] / ground[2]
    return float(X), min(float(Y), 24.0)


def draw_result(frame, point, box, world):
    x1, y1, x2, y2 = box
    X, Y = world
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
    cv2.circle(frame, tuple(map(round, point)), 5, (0, 0, 255), -1)
    label = f"X={X:.2f}m  Y={Y:.2f}m"
    for color, thickness in [((0, 0, 0), 4), ((0, 255, 0), 1)]:
        cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, thickness, cv2.LINE_AA)


def main():
    H = np.asarray(
        json.loads((FOLDER / "Homography/homography.json").read_text())["H"], dtype=float
    )
    model = YOLO(str(FOLDER / "yolo11n.pt"), verbose=False)
    cap = cv2.VideoCapture(str(FOLDER / "Video.mp4"))
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open Video.mp4")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(
            str(FOLDER / "Video_xy.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot create Video_xy.mp4")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            point, box = get_player_point(frame, model)
            if point is not None:
                world = pixel_to_world(point, H)
                if world is not None:
                    draw_result(frame, point, box, world)
            writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
