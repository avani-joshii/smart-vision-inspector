# smart-vision-inspector

A real-time OpenCV object detection tool built for CEAM's weekly "explore OpenCV" task. The core idea: every processing stage is built directly on top of the previous stage's actual output — nothing is an independent, toggleable filter. Stage 6 cannot exist without stage 5's real pixel data, and so on down the chain.

It also directly addresses the task of reducing compute dependency: the expensive per-object analysis only re-runs when motion is detected in frame, using background subtraction as a cheap gating signal.

## What it does

- Detects objects live from a webcam feed and classifies their shape (Triangle, Rectangle, Square, Pentagon, Hexagon, Circle)
- Lets you step through all 8 pipeline stages individually (keys `1`–`8`) to see exactly how the image is transformed at each point
- Shows a live **ACTIVE / IDLE** status driven by motion detection, and skips the expensive analysis step entirely when idle

## Pipeline

```
Camera → Flip → Resize → Grayscale → Gaussian Blur → Canny
       → Dilate → Find Contours → Filter + Analyze → Final Annotated Output
```

| Stage | Function(s) | Why |
|---|---|---|
| 1. Raw Frame | `flip`, `resize` | Mirrors the feed and fixes a consistent frame size for every later stage |
| 2. Grayscale | `cvtColor` | Color carries no edge info Canny needs; also cuts compute ~3x for every later stage |
| 3. Blurred | `GaussianBlur` | Removes sensor noise before edge detection, so Canny reacts to real boundaries, not noise |
| 4. Edges | `Canny` | Computes actual intensity-gradient boundaries — the real edge map everything downstream depends on |
| 5. Closed Edges | `dilate` | Canny edges are often broken into fragments; dilating bridges gaps into closed loops |
| 6. Raw Contours | `findContours` | Traces every closed loop in the edge map — unfiltered at this point |
| 7. Filtered + Analyzed | `contourArea`, `boundingRect`, `arcLength`, `approxPolyDP`, `moments`, `convexHull` | Drops noise-sized/border-touching contours, then computes real geometry for what remains |
| 8. Final Output | `drawContours`, `rectangle`, `circle`, `polylines`, `putText` | Draws bounding box, contour, hull, centroid, and shape+area label on the original frame |

**Shape classification** (`approxPolyDP` vertex count): 3 = Triangle, 4 = Rectangle/Square (via aspect ratio), 5 = Pentagon, 6 = Hexagon, 7+ = Circle.

## Compute optimization

`createBackgroundSubtractorMOG2` builds a statistical model of the empty scene and flags pixels that deviate from it as motion — a cheap, purpose-built signal computed every frame.

The expensive stage-7 analysis (geometry + shape classification for every contour) only runs when motion is above a threshold. When the scene is idle, the last computed result is reused and redrawn instead of recomputed. This is toggle-able live with `o` so gated vs always-on performance can be directly compared.

**Why this matters:** object geometry doesn't change between frames where nothing moved, so recomputing it every frame at 30fps is wasted work. Same principle as debouncing, applied to a compute pipeline instead of a button.

## Controls

| Key | Action |
|---|---|
| `1`–`8` | Jump to a specific pipeline stage |
| `←` / `→` | Step through stages one at a time |
| `o` | Toggle motion-gated compute optimization on/off |
| `s` | Save the current stage's image |
| `q` | Quit |

## Setup

```bash
pip install opencv-python numpy
python smart_vision_pipeline_stepper.py
```
## Notes / limitations

- `MIN_CONTOUR_AREA` and `MOTION_AREA_THRESHOLD` are tuned for average indoor lighting — adjust if detection feels too sensitive or not sensitive enough for your setup.
- Contours touching the frame border are excluded, since Canny can otherwise pick up the image's own edge as a false "object."
- An earlier iteration of this project attempted a document-scanner pipeline using `adaptiveThreshold` for boundary detection — abandoned after debugging showed `adaptiveThreshold` is built for local contrast separation (e.g. text on paper), not reliable whole-object outline detection. Switched to `Canny` for that role instead.
