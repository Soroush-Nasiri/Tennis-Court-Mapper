# Tennis Player Coordinates

Map a tennis player's pixel position to ground coordinates `(X, Y)` in meters using YOLO and a homography matrix. First, calibrate the mapping with five known court points, then calculate and display the player's position throughout the video.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Add your video

Place your video beside `player_xy.py`. Its filename must be exactly **`Video.mp4`**. Run the commands below from the project root.

## 2. Extract 10 frames

```bash
python extract_frames.py
```

This saves 10 roughly evenly spaced PNG frames to `frames/`, preserving the original resolution without resizing or cropping.

## 3. Select five reference points

Choose five identifiable points on the ground whose real-world coordinates are known, such as court-line intersections or net-post ground contacts. Spread them across the court; avoid placing them all on one line.

The file `Homography/reference_point_picker.html` is a fully **AI Generated** browser tool for uploading a frame, zooming in and selecting pixel coordinates. Open it in your browser, choose **Court reference points**, mark points `A` through `E`, then click **Save points JSON**. Save or replace the result as `Homography/reference_points.json`.

Each point in this JSON has two key fields:

- `pixel_uv`: original-image coordinates `(u, v)`, measured from the top-left corner.
- `world_xyz_m`: real-world coordinates `[X, Y, 0]` in meters, with `Z = 0` representing the ground plane.

The picker contains example point names and world coordinates. Edit these in the exported JSON for your own references, using one consistent ground origin and axis directions.

Keep the camera position, zoom, image resolution, crop and lens correction consistent with the reference frame. Recalibrate if this geometry changes.

## 4. Compute the homography

```bash
python Homography/compute_homography.py
```

This reads the five pixel/world pairs from `reference_points.json` and saves the **pixel-to-ground** matrix under the key `H` in `Homography/homography.json`. The included JSON files are examples; compute the mapping for your own video.

## 5. Calculate the player's position

```bash
python player_xy.py
```

YOLO11n selects the highest-confidence person detection in each frame. `get_player_point` uses the bottom center of that bounding box as an estimate of foot contact with the ground, without requiring visible ankle keypoints behind the net.

`pixel_to_world` transforms that pixel into `(X, Y)` using the saved matrix `H`, capping `Y` at 24 meters with `min(Y, 24)`. The box, red ground point and coordinates are drawn on **`Video_xy.mp4`**, saved beside the script at the original resolution and frame rate, without audio.

The model weights `yolo11n.pt` download automatically on first use if missing. None of the scripts require command-line arguments.
