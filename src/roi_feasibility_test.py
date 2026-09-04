import cv2
import time
import os
import csv
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "face_landmarker.task"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "roi_feasibility_results.csv"
)

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Size used for the ROI sent to MediaPipe.
# The original ROI is enlarged to this size.
ROI_PROCESS_WIDTH = 1280
ROI_PROCESS_HEIGHT = 720

WINDOW_NAME = "ROI Feasibility Test"


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nFace Landmarker model not found.\n\n"
        f"Expected:\n{MODEL_PATH}\n"
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    output_face_blendshapes=False,

    output_facial_transformation_matrixes=False
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    camera.release()

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )


if not camera.isOpened():

    raise RuntimeError(
        "Could not open the camera."
    )


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

try:

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

except Exception:

    pass


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def draw_text(
    image,
    text,
    position,
    scale=0.55,
    color=(255, 255, 255),
    thickness=1
):

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_landmarks(
    image,
    landmarks,
    width,
    height
):

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        if (
            0 <= x < width
            and
            0 <= y < height
        ):

            cv2.circle(
                image,
                (x, y),
                1,
                (255, 255, 255),
                -1
            )


def get_face_box(
    landmarks,
    width,
    height
):

    xs = []
    ys = []

    for landmark in landmarks:

        xs.append(
            landmark.x * width
        )

        ys.append(
            landmark.y * height
        )

    if not xs:

        return None

    x1 = max(
        0,
        int(min(xs))
    )

    y1 = max(
        0,
        int(min(ys))
    )

    x2 = min(
        width - 1,
        int(max(xs))
    )

    y2 = min(
        height - 1,
        int(max(ys))
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# OPEN CAMERA AND SELECT ROI
# ============================================================

print()
print("=" * 70)
print("SLEEPING EYE DETECTION")
print("ROI + RESIZE FEASIBILITY TEST")
print("=" * 70)
print()
print("First you will select the BED / PILLOW region.")
print()
print("Select a rectangle around the area where the")
print("sleeping person's face is expected to appear.")
print()
print("After selecting the ROI:")
print("  ENTER / SPACE = confirm")
print("  C = cancel selection")
print()
print("=" * 70)
print()


# ============================================================
# GET INITIAL FRAME
# ============================================================

success, first_frame = camera.read()

if not success:

    camera.release()

    raise RuntimeError(
        "Could not read the camera."
    )


first_frame = cv2.flip(
    first_frame,
    1
)


# ============================================================
# SELECT ROI
# ============================================================

print(
    "Select the BED / PILLOW region with your mouse..."
)

roi = cv2.selectROI(
    "Select Bed / Pillow ROI",
    first_frame,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyWindow(
    "Select Bed / Pillow ROI"
)


roi_x, roi_y, roi_w, roi_h = roi


if roi_w <= 0 or roi_h <= 0:

    camera.release()

    cv2.destroyAllWindows()

    raise RuntimeError(
        "No ROI was selected."
    )


print()
print("ROI selected:")
print(
    f"X={roi_x}, Y={roi_y}, "
    f"Width={roi_w}, Height={roi_h}"
)
print()


# ============================================================
# CSV
# ============================================================

csv_exists = os.path.exists(
    RESULTS_CSV
)

csv_file = open(
    RESULTS_CSV,
    "a",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(
    csv_file
)

if not csv_exists:

    csv_writer.writerow([
        "timestamp",
        "mode",
        "face_detected",
        "landmark_count",
        "face_width_px",
        "face_height_px",
        "roi_width_px",
        "roi_height_px",
        "fps"
    ])


# ============================================================
# TEST STATE
# ============================================================

mode = "ROI_RESIZED"

show_landmarks = True

last_timestamp = -1

frame_count = 0

fps = 0.0

fps_start = time.monotonic()

total_frames = 0

full_detected_frames = 0

roi_detected_frames = 0


# ============================================================
# MEDIAPIPE
# ============================================================

try:

    with FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        print("=" * 70)
        print("ROI TEST STARTED")
        print("=" * 70)
        print()
        print("Controls:")
        print()
        print("F = Full frame detection")
        print("R = ROI + resized detection")
        print("D = Toggle landmarks")
        print("Q = Quit")
        print()
        print(
            "For the main experiment, move FAR from the camera."
        )
        print(
            "Then compare F and R."
        )
        print()
        print("=" * 70)
        print()

        while True:

            # ====================================================
            # CAMERA
            # ====================================================

            success, frame = camera.read()

            if not success:

                continue

            frame = cv2.flip(
                frame,
                1
            )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            # ====================================================
            # FPS
            # ====================================================

            now = time.monotonic()

            frame_count += 1

            elapsed = (
                now - fps_start
            )

            if elapsed >= 1.0:

                fps = (
                    frame_count /
                    elapsed
                )

                frame_count = 0

                fps_start = now

            # ====================================================
            # DISPLAY
            # ====================================================

            display = frame.copy()

            # Draw original ROI
            cv2.rectangle(
                display,
                (
                    roi_x,
                    roi_y
                ),
                (
                    roi_x + roi_w,
                    roi_y + roi_h
                ),
                (255, 200, 0),
                2
            )

            draw_text(
                display,
                "BED / PILLOW ROI",
                (
                    roi_x,
                    max(
                        25,
                        roi_y - 10
                    )
                ),
                0.5,
                (255, 200, 0),
                1
            )

            # ====================================================
            # SELECT IMAGE FOR MEDIAPIPE
            # ====================================================

            if mode == "ROI_RESIZED":

                # ------------------------------------------------
                # Crop ROI
                # ------------------------------------------------

                roi_frame = frame[
                    roi_y:roi_y + roi_h,
                    roi_x:roi_x + roi_w
                ]

                if roi_frame.size == 0:

                    continue

                # ------------------------------------------------
                # Resize ROI
                # ------------------------------------------------

                process_frame = cv2.resize(
                    roi_frame,
                    (
                        ROI_PROCESS_WIDTH,
                        ROI_PROCESS_HEIGHT
                    ),
                    interpolation=cv2.INTER_CUBIC
                )

            else:

                # Full frame
                process_frame = frame

            # ====================================================
            # RGB
            # ====================================================

            rgb = cv2.cvtColor(
                process_frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb
            )

            # ====================================================
            # TIMESTAMP
            # ====================================================

            timestamp_ms = int(
                now * 1000
            )

            if timestamp_ms <= last_timestamp:

                timestamp_ms = (
                    last_timestamp + 1
                )

            last_timestamp = timestamp_ms

            # ====================================================
            # DETECT
            # ====================================================

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            total_frames += 1

            face_detected = False

            landmark_count = 0

            face_width = 0

            face_height = 0

            # ====================================================
            # FACE DETECTED
            # ====================================================

            if result.face_landmarks:

                face_detected = True

                landmarks = (
                    result.face_landmarks[0]
                )

                landmark_count = len(
                    landmarks
                )

                if mode == "ROI_RESIZED":

                    # --------------------------------------------
                    # Face box is in resized ROI coordinates.
                    # Convert it back to original ROI coordinates.
                    # --------------------------------------------

                    box = get_face_box(
                        landmarks,
                        ROI_PROCESS_WIDTH,
                        ROI_PROCESS_HEIGHT
                    )

                    if box:

                        x1, y1, x2, y2 = box

                        scale_x = (
                            roi_w /
                            ROI_PROCESS_WIDTH
                        )

                        scale_y = (
                            roi_h /
                            ROI_PROCESS_HEIGHT
                        )

                        original_x1 = int(
                            roi_x +
                            x1 * scale_x
                        )

                        original_y1 = int(
                            roi_y +
                            y1 * scale_y
                        )

                        original_x2 = int(
                            roi_x +
                            x2 * scale_x
                        )

                        original_y2 = int(
                            roi_y +
                            y2 * scale_y
                        )

                        face_width = (
                            original_x2 -
                            original_x1
                        )

                        face_height = (
                            original_y2 -
                            original_y1
                        )

                        # Draw result on original frame
                        cv2.rectangle(
                            display,
                            (
                                original_x1,
                                original_y1
                            ),
                            (
                                original_x2,
                                original_y2
                            ),
                            (0, 255, 0),
                            2
                        )

                    # --------------------------------------------
                    # Draw landmarks on display
                    # --------------------------------------------

                    if show_landmarks:

                        for landmark in landmarks:

                            px = int(
                                landmark.x *
                                ROI_PROCESS_WIDTH
                            )

                            py = int(
                                landmark.y *
                                ROI_PROCESS_HEIGHT
                            )

                            if (
                                0 <= px <
                                ROI_PROCESS_WIDTH
                                and
                                0 <= py <
                                ROI_PROCESS_HEIGHT
                            ):

                                original_px = int(
                                    roi_x +
                                    px * (
                                        roi_w /
                                        ROI_PROCESS_WIDTH
                                    )
                                )

                                original_py = int(
                                    roi_y +
                                    py * (
                                        roi_h /
                                        ROI_PROCESS_HEIGHT
                                    )
                                )

                                if (
                                    0 <= original_px <
                                    frame_width
                                    and
                                    0 <= original_py <
                                    frame_height
                                ):

                                    cv2.circle(
                                        display,
                                        (
                                            original_px,
                                            original_py
                                        ),
                                        1,
                                        (255, 255, 255),
                                        -1
                                    )

                else:

                    # --------------------------------------------
                    # Full-frame face box
                    # --------------------------------------------

                    box = get_face_box(
                        landmarks,
                        frame_width,
                        frame_height
                    )

                    if box:

                        x1, y1, x2, y2 = box

                        face_width = (
                            x2 - x1
                        )

                        face_height = (
                            y2 - y1
                        )

                        cv2.rectangle(
                            display,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                    # --------------------------------------------
                    # Full-frame landmarks
                    # --------------------------------------------

                    if show_landmarks:

                        draw_landmarks(
                            display,
                            landmarks,
                            frame_width,
                            frame_height
                        )

            # ====================================================
            # STATISTICS
            # ====================================================

            if mode == "ROI_RESIZED":

                if face_detected:

                    roi_detected_frames += 1

            else:

                if face_detected:

                    full_detected_frames += 1

            # ====================================================
            # DETECTION RATE
            # ====================================================

            if total_frames > 0:

                if mode == "ROI_RESIZED":

                    detection_rate = (
                        roi_detected_frames /
                        total_frames
                    ) * 100

                else:

                    detection_rate = (
                        full_detected_frames /
                        total_frames
                    ) * 100

            else:

                detection_rate = 0.0

            # ====================================================
            # UI
            # ====================================================

            if face_detected:

                draw_text(
                    display,
                    "FACE DETECTED",
                    (20, 35),
                    0.75,
                    (0, 255, 0),
                    2
                )

            else:

                draw_text(
                    display,
                    "NO FACE DETECTED",
                    (20, 35),
                    0.75,
                    (0, 0, 255),
                    2
                )

            draw_text(
                display,
                f"MODE: {mode}",
                (20, 70),
                0.55,
                (0, 255, 255),
                2
            )

            draw_text(
                display,
                f"Landmarks: {landmark_count}",
                (20, 105),
                0.50,
                (255, 255, 255),
                1
            )

            draw_text(
                display,
                f"FPS: {fps:.1f}",
                (20, 135),
                0.50,
                (255, 255, 255),
                1
            )

            draw_text(
                display,
                f"Face width: {face_width}px",
                (20, 165),
                0.50,
                (255, 255, 255),
                1
            )

            draw_text(
                display,
                f"Face height: {face_height}px",
                (20, 195),
                0.50,
                (255, 255, 255),
                1
            )

            draw_text(
                display,
                f"Detection: {detection_rate:.1f}%",
                (20, 230),
                0.52,
                (0, 255, 255),
                1
            )

            draw_text(
                display,
                (
                    f"ROI: "
                    f"{roi_w} x {roi_h}px"
                ),
                (20, 260),
                0.48,
                (255, 200, 0),
                1
            )

            draw_text(
                display,
                (
                    "F = Full frame | "
                    "R = ROI + Resize"
                ),
                (20, frame_height - 95),
                0.45,
                (220, 220, 220),
                1
            )

            draw_text(
                display,
                (
                    "D = Landmarks | "
                    "Q = Quit"
                ),
                (20, frame_height - 65),
                0.45,
                (220, 220, 220),
                1
            )

            # ====================================================
            # SHOW
            # ====================================================

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            # ====================================================
            # CSV
            # ====================================================

            csv_writer.writerow([

                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                mode,

                int(face_detected),

                landmark_count,

                face_width,

                face_height,

                roi_w,

                roi_h,

                round(
                    fps,
                    2
                )
            ])

            if total_frames % 30 == 0:

                csv_file.flush()

            # ====================================================
            # KEYBOARD
            # ====================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ----------------------------------------------------
            # FULL FRAME
            # ----------------------------------------------------

            if key == ord("f"):

                mode = "FULL_FRAME"

                total_frames = 0

                full_detected_frames = 0

                roi_detected_frames = 0

                print()
                print(
                    "MODE: FULL FRAME"
                )
                print(
                    "Move FAR from the camera and observe."
                )

            # ----------------------------------------------------
            # ROI + RESIZE
            # ----------------------------------------------------

            elif key == ord("r"):

                mode = "ROI_RESIZED"

                total_frames = 0

                full_detected_frames = 0

                roi_detected_frames = 0

                print()
                print(
                    "MODE: ROI + RESIZE"
                )
                print(
                    "Keep the same distance and position."
                )

            # ----------------------------------------------------
            # LANDMARK TOGGLE
            # ----------------------------------------------------

            elif key == ord("d"):

                show_landmarks = (
                    not show_landmarks
                )

                print(
                    "Landmarks:",
                    "ON"
                    if show_landmarks
                    else
                    "OFF"
                )

            # ----------------------------------------------------
            # QUIT
            # ----------------------------------------------------

            elif key == ord("q"):

                break


finally:

    camera.release()

    cv2.destroyAllWindows()

    csv_file.flush()

    csv_file.close()


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("ROI FEASIBILITY TEST COMPLETE")
print("=" * 70)
print()
print(
    "Results saved to:"
)
print(
    RESULTS_CSV
)
print()
print("=" * 70)