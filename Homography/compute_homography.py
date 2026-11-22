import json
from pathlib import Path

import cv2
import numpy as np

folder = Path(__file__).resolve().parent
with (folder / "reference_points.json").open() as file:
    data = json.load(file)

pixels = np.array([p["pixel_uv"] for p in data["points"]], dtype=float)
ground = np.array([p["world_xyz_m"][:2] for p in data["points"]], dtype=float)

H, _ = cv2.findHomography(pixels, ground, method=0)

with (folder / "homography.json").open("w") as file:
    json.dump({"H": H.tolist()}, file, indent=2)
