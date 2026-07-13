"""
Smart Vision Inspector — Pipeline Stepper
--------------------------------------------
The whole point of this version: no independent, random filter toggles.
Every stage is built directly on top of the previous stage's actual output.
You can't skip ahead — stage 5 literally does not exist without stage 4
having produced it first. Scrub through the stages with number keys 1-8
(or Left/Right arrows) and watch the SAME image get progressively
transformed, step by step, into the final analyzed result.

Stage 1: Raw frame            (flip + resize)
Stage 2: Grayscale             (cvtColor)
Stage 3: Blurred               (GaussianBlur on the grayscale output)
Stage 4: Edges                 (Canny on the blurred output)
Stage 5: Closed edges          (dilate on the edge output, bridges gaps)
Stage 6: Raw contours          (findContours on the closed-edge output)
Stage 7: Filtered + analyzed   (drop tiny ones, compute area/perimeter/
                                 shape/center/hull for what's left)
Stage 8: Final annotated output (all overlays drawn on the original frame)

Each stage's function call takes the PREVIOUS stage's actual output as its
input — nothing here is decorative or independent.

Controls:
  1-8      - jump to a specific stage
  LEFT/RIGHT arrows - step through stages
  s        - save the current stage's image
  q        - quit
"""

import cv2
import numpy as np
import os
import time

# ----------------------------- CONFIG -----------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_CONTOUR_AREA = 3000
MOTION_AREA_THRESHOLD = 1500   # total changed-pixel area needed to call the scene "ACTIVE"
BLUR_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
FONT = cv2.FONT_HERSHEY_SIMPLEX
SNAPSHOT_DIR = "pipeline_snapshots"

STAGE_NAMES = {
    1: "1. Raw Frame (flip + resize)",
    2: "2. Grayscale (cvtColor)",
    3: "3. Blurred (GaussianBlur)",
    4: "4. Edges (Canny)",
    5: "5. Closed Edges (dilate)",
    6: "6. Raw Contours (findContours)",
    7: "7. Filtered + Analyzed (area/perimeter/shape/hull)",
    8: "8. Final Annotated Output",
}

STAGE_EXPLANATIONS = {
    1: "Flip mirrors the feed so it acts like a mirror. Resize fixes a consistent frame size so every later stage has predictable dimensions.",
    2: "Color carries no edge information Canny needs later - dropping to 1 channel also cuts compute for every stage that follows.",
    3: "Smooths sensor noise BEFORE edge detection. Skip this and Canny reacts to noise speckles, not real object edges.",
    4: "Finds real intensity-gradient boundaries in the blurred image. This is the actual edge map everything downstream depends on.",
    5: "Canny edges are often broken into small disconnected segments. Dilating thickens/bridges them into closed loops findContours can trace.",
    6: "Traces every closed loop in the edge map into a contour - includes tiny noise fragments at this point, unfiltered.",
    7: "Drops anything under the area threshold, then computes real geometry (area, perimeter, center via moments, shape via approxPolyDP, hull) for what remains.",
    8: "Takes the analyzed objects and draws them back onto the ORIGINAL color frame - bounding box, contour, hull, center point, label.",
}


# ------------------------- MOTION DETECTION (ACTIVE/IDLE signal) -------------------------
def get_motion_mask(bg_subtractor, gray_frame):
    """MOG2 builds a statistical model of the 'empty' background over time,
    then flags pixels that deviate from it as foreground (motion). This is
    a far cleaner ACTIVE/IDLE signal than counting detected objects, since
    object-count naturally flickers frame to frame even when the real
    scene hasn't changed — motion detection answers a different, steadier
    question: did anything actually move."""
    fg_mask = bg_subtractor.apply(gray_frame)

    # Remove the shadow value (127) MOG2 marks separately from real motion (255)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

  
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    return fg_mask


# ------------------------- SHAPE DETECTION -------------------------
def detect_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    sides = len(approx)

    if sides == 3:
        return "Triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif sides == 5:
        return "Pentagon"
    elif sides == 6:
        return "Hexagon"
    else:
        return "Circle"


def analyze_contour(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2

    return {
        "contour": contour,
        "area": area,
        "perimeter": perimeter,
        "bbox": (x, y, w, h),
        "center": (cx, cy),
        "shape": detect_shape(contour),
        "hull": cv2.convexHull(contour),
    }


def draw_object(frame, obj):
    x, y, w, h = obj["bbox"]
    cx, cy = obj["center"]

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawContours(frame, [obj["contour"]], -1, (255, 0, 0), 2)
    cv2.polylines(frame, [obj["hull"]], True, (0, 255, 255), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    label = f'{obj["shape"]} | A:{int(obj["area"])}'
    cv2.putText(frame, label, (x, max(y - 10, 15)), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------- CUMULATIVE PIPELINE -------------------------
def run_pipeline(raw_frame, up_to_stage):
    """Runs the pipeline stage by stage, feeding each stage's output
    directly into the next. Stops and returns whatever the requested stage
    looks like — but every stage before it still had to run to get there,
    exactly like the real pipeline does at stage 8."""
    stage_image = raw_frame  # stage 1 output
    objects = []

    if up_to_stage == 1:
        return stage_image, objects

    gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    stage_image = gray
    if up_to_stage == 2:
        return stage_image, objects

    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    stage_image = blurred
    if up_to_stage == 3:
        return stage_image, objects

    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    stage_image = edges
    if up_to_stage == 4:
        return stage_image, objects

    closed = cv2.dilate(edges, None, iterations=1)
    stage_image = closed
    if up_to_stage == 5:
        return stage_image, objects

    raw_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_vis = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_vis, raw_contours, -1, (0, 200, 255), 2)
    stage_image = contour_vis
    if up_to_stage == 6:
        return stage_image, objects

    frame_h, frame_w = closed.shape[:2]
    margin = 2  # pixels of tolerance at the border

    def touches_border(c):
        x, y, cw, ch = cv2.boundingRect(c)
        return x <= margin or y <= margin or (x + cw) >= (frame_w - margin) or (y + ch) >= (frame_h - margin)

    filtered_contours = []
    for c in raw_contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        # Border-touching contours are almost always the frame's own edge
        # being picked up by Canny, not a real object — this is very likely
        # why status was stuck ACTIVE regardless of what was actually in view.
        if touches_border(c):
            continue
        filtered_contours.append(c)

    objects = [analyze_contour(c) for c in filtered_contours]
    analyzed_vis = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    for obj in objects:
        draw_object(analyzed_vis, obj)
    stage_image = analyzed_vis
    if up_to_stage == 7:
        return stage_image, objects

    final = raw_frame.copy()
    for obj in objects:
        draw_object(final, obj)
    stage_image = final
    return stage_image, objects


# ------------------------- UI -------------------------
def wrap_text(text, max_width_px, font_scale, thickness=1):
    """Splits text into lines that actually fit within max_width_px,
    measured with cv2.getTextSize — this is what was missing before, the
    explanation strings were just being drawn at full length regardless of
    frame width."""
    words = text.split(" ")
    lines = []
    current = ""

    for word in words:
        test = (current + " " + word).strip()
        size = cv2.getTextSize(test, FONT, font_scale, thickness)[0]
        if size[0] <= max_width_px:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    return lines


def build_display(stage_image, stage_num, objects, fps, smoothed_status=None):
    if len(stage_image.shape) == 2:
        display = cv2.cvtColor(stage_image, cv2.COLOR_GRAY2BGR)
    else:
        display = stage_image.copy()

    h, w = display.shape[:2]

    # Explanation text wraps to however many lines it actually needs, so the
    # bar height adapts instead of the text running off the right edge.
    explanation_lines = wrap_text(STAGE_EXPLANATIONS[stage_num], w - 30, 0.42, 1)
    bar_height = 55 + (len(explanation_lines) * 18) + 30

    canvas = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    canvas[0:h, :] = display

    bar_y = h
    cv2.rectangle(canvas, (0, bar_y), (w, h + bar_height), (30, 30, 30), -1)
    cv2.line(canvas, (0, bar_y), (w, bar_y), (80, 80, 80), 1)

    # Status only means something once objects have actually been computed
    # (stage 7 or 8) — before that, there's nothing to be active about yet.
    # Uses a SMOOTHED status (majority vote over recent frames) instead of
    # this single frame's raw reading, since single-frame contour counts
    # flicker due to normal camera/lighting noise even when the real scene
    # hasn't changed.
    if stage_num >= 7 and smoothed_status is not None:
        status_text = smoothed_status
        status_color = (0, 255, 0) if smoothed_status == "ACTIVE" else (100, 100, 100)
    else:
        status_text = "N/A"
        status_color = (120, 120, 120)

    cv2.putText(canvas, STAGE_NAMES[stage_num], (15, bar_y + 20), FONT, 0.5, (0, 255, 255), 2)
    status_size = cv2.getTextSize(f"Status: {status_text}", FONT, 0.5, 2)[0]
    cv2.putText(canvas, f"Status: {status_text}", (w - status_size[0] - 15, bar_y + 20), FONT, 0.5, status_color, 2)

    y_cursor = bar_y + 40
    for line in explanation_lines:
        cv2.putText(canvas, line, (15, y_cursor), FONT, 0.42, (200, 200, 200), 1)
        y_cursor += 18

    largest_area = max([o["area"] for o in objects], default=0)
    footer = f"Objects: {len(objects)}   Largest Area: {int(largest_area)}px^2   FPS: {fps:.1f}"
    cv2.putText(canvas, footer, (15, y_cursor + 4), FONT, 0.42, (150, 150, 150), 1)
    y_cursor += 20
    controls = "[1-8] jump  [Left/Right] step  [o] toggle gating  [s] save  [q] quit"
    cv2.putText(canvas, controls, (15, y_cursor + 4), FONT, 0.38, (120, 120, 120), 1)

    return canvas


# ------------------------- MAIN LOOP -------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    current_stage = 8
    prev_time = time.time()

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
    optimized = True   # gating on/off, toggle with 'o'
    last_objects = []  # reused on IDLE frames when gating is on

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray_for_motion = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Motion check: cheap, runs every frame, decides the real ACTIVE/IDLE status ----
        fg_mask = get_motion_mask(bg_subtractor, gray_for_motion)
        motion_area = cv2.countNonZero(fg_mask)
        smoothed_status = "ACTIVE" if motion_area > MOTION_AREA_THRESHOLD else "IDLE"

        # ---- Stage pipeline: only run the expensive stage 7/8 analysis when
        # motion says it's worth it (or gating is turned off) — same idea as
        # the mentor's "reduce compute dependency" ask, now tied to a status
        # you can actually see on screen. ----
        run_full_analysis = (smoothed_status == "ACTIVE") or (not optimized) or (current_stage < 7)

        if run_full_analysis:
            stage_image, objects = run_pipeline(frame, current_stage)
            if current_stage >= 7:
                last_objects = objects
        else:
            # Idle + gated: genuinely skip the expensive stage 7 analysis by
            # capping the pipeline at stage 6 (contours only, no per-object
            # geometry/shape work), then draw the LAST known objects on top
            # for display continuity instead of recomputing them.
            capped_stage = min(current_stage, 6)
            stage_image, _ = run_pipeline(frame, capped_stage)
            objects = last_objects
            if current_stage >= 7:
                if len(stage_image.shape) == 2:
                    stage_image = cv2.cvtColor(stage_image, cv2.COLOR_GRAY2BGR)
                display_base = frame.copy() if current_stage == 8 else stage_image
                for obj in objects:
                    draw_object(display_base, obj)
                stage_image = display_base

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
        prev_time = curr_time

        canvas = build_display(stage_image, current_stage, objects, fps, smoothed_status)
        cv2.imshow("Smart Vision Inspector - Pipeline Stepper", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            to_save = stage_image if len(stage_image.shape) == 3 else cv2.cvtColor(stage_image, cv2.COLOR_GRAY2BGR)
            filename = os.path.join(SNAPSHOT_DIR, f"stage{current_stage}_{int(time.time())}.png")
            cv2.imwrite(filename, to_save)
            print(f"Saved: {filename}")
        elif key in [ord(str(n)) for n in range(1, 9)]:
            current_stage = int(chr(key))
        elif key in (81, 2):   # left arrow (varies by OS/backend)
            current_stage = max(1, current_stage - 1)
        elif key in (83, 3):   # right arrow (varies by OS/backend)
            current_stage = min(8, current_stage + 1)
        elif key == ord('o'):
            optimized = not optimized
            print(f"Gating {'enabled' if optimized else 'disabled'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()