import cv2
import time
import os
import csv
import math
import mediapipe as mp


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "face_landmarker.task"
)

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

WINDOW_NAME = "Sleeping Eye | Feasibility Test"

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

FRAME_CSV = os.path.join(
    OUTPUT_DIR,
    "feasibility_results.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "feasibility_summary.csv"
)


# ============================================================
# TEST DEFINITIONS
# ============================================================

TESTS = {

    "1": "FRONT",

    "2": "HEAD_LEFT",

    "3": "HEAD_RIGHT",

    "4": "TILT_LEFT",

    "5": "TILT_RIGHT",

    "6": "LOOK_UP",

    "7": "LOOK_DOWN",

    "8": "NEAR_CAMERA",

    "9": "FAR_CAMERA"
}


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nFace Landmarker model not found.\n\n"
        f"Expected:\n{MODEL_PATH}\n"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CSV HEADERS
# ============================================================

FRAME_HEADERS = [

    "timestamp",

    "test_id",

    "test_name",

    "face_detected",

    "landmark_count",

    "face_width_px",

    "face_height_px",

    "face_area_percent",

    "face_center_x_percent",

    "face_center_y_percent",

    "yaw_deg",

    "pitch_deg",

    "roll_deg",

    "fps"
]


SUMMARY_HEADERS = [

    "test_id",

    "test_name",

    "total_frames",

    "face_detected_frames",

    "detection_rate_percent",

    "landmark_success_rate_percent",

    "average_fps",

    "average_face_width_px",

    "average_face_height_px",

    "average_face_area_percent",

    "average_yaw_deg",

    "average_pitch_deg",

    "average_roll_deg"
]


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


# ============================================================
# GET FACE BOUNDING BOX
# ============================================================

def get_face_box(
    landmarks,
    width,
    height
):

    xs = []
    ys = []

    for landmark in landmarks:

        if not math.isfinite(landmark.x):
            continue

        if not math.isfinite(landmark.y):
            continue

        xs.append(
            landmark.x * width
        )

        ys.append(
            landmark.y * height
        )

    if not xs or not ys:

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
# HEAD POSE ESTIMATION
# ============================================================

def estimate_head_pose(
    landmarks,
    width,
    height
):

    """
    Estimate approximate head orientation.

    Returns:
        yaw
        pitch
        roll

    IMPORTANT:
    These values are approximate because we are using
    an approximate camera matrix rather than a calibrated
    physical camera.
    """

    try:

        # ----------------------------------------------------
        # MediaPipe landmark indices
        # ----------------------------------------------------

        # Nose tip
        nose = landmarks[1]

        # Chin
        chin = landmarks[152]

        # Left eye outer corner
        left_eye = landmarks[263]

        # Right eye outer corner
        right_eye = landmarks[33]

        # Left mouth corner
        left_mouth = landmarks[291]

        # Right mouth corner
        right_mouth = landmarks[61]

        image_points = [

            (
                nose.x * width,
                nose.y * height
            ),

            (
                chin.x * width,
                chin.y * height
            ),

            (
                left_eye.x * width,
                left_eye.y * height
            ),

            (
                right_eye.x * width,
                right_eye.y * height
            ),

            (
                left_mouth.x * width,
                left_mouth.y * height
            ),

            (
                right_mouth.x * width,
                right_mouth.y * height
            )
        ]

        image_points = (
            cv2
            .Mat
            if False
            else
            image_points
        )

        image_points = cv2.UMat(
            image_points
        ).get().astype(
            "float64"
        )

        # ----------------------------------------------------
        # Generic 3D face model
        # ----------------------------------------------------

        model_points = [

            (0.0, 0.0, 0.0),          # nose

            (0.0, -63.6, -12.5),      # chin

            (-43.3, 32.7, -26.0),     # left eye

            (43.3, 32.7, -26.0),      # right eye

            (-28.9, -28.9, -24.1),    # left mouth

            (28.9, -28.9, -24.1)      # right mouth
        ]

        model_points = (
            cv2.UMat(
                model_points
            ).get().astype(
                "float64"
            )
        )

        # ----------------------------------------------------
        # Approximate camera matrix
        # ----------------------------------------------------

        focal_length = float(width)

        center = (
            width / 2.0,
            height / 2.0
        )

        camera_matrix = cv2.UMat(
            [
                [
                    focal_length,
                    0,
                    center[0]
                ],
                [
                    0,
                    focal_length,
                    center[1]
                ],
                [
                    0,
                    0,
                    1
                ]
            ]
        ).get().astype(
            "float64"
        )

        distortion = cv2.UMat(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0]
            ]
        ).get().astype(
            "float64"
        )

        # ----------------------------------------------------
        # Solve PnP
        # ----------------------------------------------------

        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        )

        if not success:

            return None

        # ----------------------------------------------------
        # Rotation matrix
        # ----------------------------------------------------

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        # ----------------------------------------------------
        # Euler angles
        # ----------------------------------------------------

        sy = math.sqrt(
            rotation_matrix[0, 0] ** 2 +
            rotation_matrix[1, 0] ** 2
        )

        singular = sy < 1e-6

        if not singular:

            pitch = math.atan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2]
            )

            yaw = math.atan2(
                -rotation_matrix[2, 0],
                sy
            )

            roll = math.atan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0]
            )

        else:

            pitch = math.atan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1]
            )

            yaw = math.atan2(
                -rotation_matrix[2, 0],
                sy
            )

            roll = 0

        pitch = math.degrees(
            pitch
        )

        yaw = math.degrees(
            yaw
        )

        roll = math.degrees(
            roll
        )

        return (
            yaw,
            pitch,
            roll
        )

    except Exception:

        return None


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
    FRAME_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

camera.set(
    cv2.CAP_PROP_FPS,
    30
)

try:

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

except Exception:

    pass


# ============================================================
# MEDIAPIPE
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
# CSV FILES
# ============================================================

frame_file = open(
    FRAME_CSV,
    "w",
    newline="",
    encoding="utf-8"
)

summary_file = open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8"
)

frame_writer = csv.writer(
    frame_file
)

summary_writer = csv.writer(
    summary_file
)

frame_writer.writerow(
    FRAME_HEADERS
)

summary_writer.writerow(
    SUMMARY_HEADERS
)


# ============================================================
# TEST STATE
# ============================================================

current_test_id = "1"

current_test_name = TESTS[
    current_test_id
]

test_start_time = time.monotonic()

last_timestamp = -1

draw_landmarks = True


# ============================================================
# TEST STATISTICS
# ============================================================

stats = {}


def create_stats():

    return {

        "total_frames": 0,

        "face_detected_frames": 0,

        "landmark_success_frames": 0,

        "fps_values": [],

        "face_widths": [],

        "face_heights": [],

        "face_areas": [],

        "yaws": [],

        "pitches": [],

        "rolls": []
    }


stats[current_test_id] = (
    create_stats()
)


# ============================================================
# RESET CURRENT TEST
# ============================================================

def reset_test(
    test_id
):

    global current_test_id
    global current_test_name
    global test_start_time

    current_test_id = test_id

    current_test_name = TESTS[
        test_id
    ]

    test_start_time = (
        time.monotonic()
    )

    if test_id not in stats:

        stats[test_id] = (
            create_stats()
        )

    print()
    print(
        "=" * 65
    )

    print(
        f"TEST STARTED: "
        f"{test_id} - "
        f"{current_test_name}"
    )

    print(
        "Keep this position for approximately 15 seconds."
    )

    print(
        "=" * 65
    )


# ============================================================
# SAVE TEST SUMMARY
# ============================================================

def save_test_summary(
    test_id
):

    if test_id not in stats:

        return

    data = stats[
        test_id
    ]

    total = data[
        "total_frames"
    ]

    detected = data[
        "face_detected_frames"
    ]

    landmarks = data[
        "landmark_success_frames"
    ]

    if total == 0:

        return

    detection_rate = (
        detected /
        total
    ) * 100

    landmark_rate = (
        landmarks /
        total
    ) * 100

    average_fps = (
        sum(data["fps_values"]) /
        len(data["fps_values"])
        if data["fps_values"]
        else
        0
    )

    average_width = (
        sum(data["face_widths"]) /
        len(data["face_widths"])
        if data["face_widths"]
        else
        0
    )

    average_height = (
        sum(data["face_heights"]) /
        len(data["face_heights"])
        if data["face_heights"]
        else
        0
    )

    average_area = (
        sum(data["face_areas"]) /
        len(data["face_areas"])
        if data["face_areas"]
        else
        0
    )

    average_yaw = (
        sum(data["yaws"]) /
        len(data["yaws"])
        if data["yaws"]
        else
        0
    )

    average_pitch = (
        sum(data["pitches"]) /
        len(data["pitches"])
        if data["pitches"]
        else
        0
    )

    average_roll = (
        sum(data["rolls"]) /
        len(data["rolls"])
        if data["rolls"]
        else
        0
    )

    summary_writer.writerow([

        test_id,

        TESTS[test_id],

        total,

        detected,

        round(
            detection_rate,
            2
        ),

        round(
            landmark_rate,
            2
        ),

        round(
            average_fps,
            2
        ),

        round(
            average_width,
            2
        ),

        round(
            average_height,
            2
        ),

        round(
            average_area,
            2
        ),

        round(
            average_yaw,
            2
        ),

        round(
            average_pitch,
            2
        ),

        round(
            average_roll,
            2
        )
    ])

    summary_file.flush()

    print()
    print(
        f"TEST {test_id} - "
        f"{TESTS[test_id]}"
    )

    print(
        f"Frames: {total}"
    )

    print(
        f"Face detection: "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Landmark success: "
        f"{landmark_rate:.2f}%"
    )

    print(
        f"Average FPS: "
        f"{average_fps:.2f}"
    )

    print(
        f"Average face size: "
        f"{average_width:.1f} x "
        f"{average_height:.1f}px"
    )

    print(
        f"Average face area: "
        f"{average_area:.2f}%"
    )

    print(
        f"Average Yaw: "
        f"{average_yaw:.2f}°"
    )

    print(
        f"Average Pitch: "
        f"{average_pitch:.2f}°"
    )

    print(
        f"Average Roll: "
        f"{average_roll:.2f}°"
    )

    print()


# ============================================================
# CONSOLE INSTRUCTIONS
# ============================================================

print()
print("=" * 70)
print("SLEEPING EYE DETECTION - FEASIBILITY TEST")
print("=" * 70)
print()
print("TEST KEYS")
print()
print("1 = FRONT")
print("2 = HEAD LEFT")
print("3 = HEAD RIGHT")
print("4 = TILT LEFT")
print("5 = TILT RIGHT")
print("6 = LOOK UP")
print("7 = LOOK DOWN")
print("8 = NEAR CAMERA")
print("9 = FAR CAMERA")
print()
print("D = Toggle landmarks ON/OFF")
print("S = Save current test summary")
print("Q = Quit")
print()
print("Keep each test position for approximately 15 seconds.")
print()
print("=" * 70)
print()


# ============================================================
# FPS
# ============================================================

fps = 0.0

fps_counter = 0

fps_start = time.monotonic()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    with FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            # ====================================================
            # CAMERA FRAME
            # ====================================================

            success, frame = camera.read()

            if not success:

                continue

            # Mirror image
            frame = cv2.flip(
                frame,
                1
            )

            height, width = (
                frame.shape[:2]
            )

            # ====================================================
            # FPS
            # ====================================================

            now = time.monotonic()

            fps_counter += 1

            fps_elapsed = (
                now -
                fps_start
            )

            if fps_elapsed >= 1.0:

                fps = (
                    fps_counter /
                    fps_elapsed
                )

                fps_counter = 0

                fps_start = now

            # ====================================================
            # MEDIAPIPE IMAGE
            # ====================================================

            rgb = cv2.cvtColor(
                frame,
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

            last_timestamp = (
                timestamp_ms
            )

            # ====================================================
            # FACE LANDMARK DETECTION
            # ====================================================

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            # ====================================================
            # DEFAULT VALUES
            # ====================================================

            face_detected = False

            landmark_count = 0

            face_width = 0

            face_height = 0

            face_area_percent = 0.0

            center_x_percent = 0.0

            center_y_percent = 0.0

            yaw = None

            pitch = None

            roll = None

            display = frame.copy()

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

                # ------------------------------------------------
                # FACE BOX
                # ------------------------------------------------

                box = get_face_box(
                    landmarks,
                    width,
                    height
                )

                if box:

                    x1, y1, x2, y2 = box

                    face_width = (
                        x2 - x1
                    )

                    face_height = (
                        y2 - y1
                    )

                    face_area = (
                        face_width *
                        face_height
                    )

                    total_area = (
                        width *
                        height
                    )

                    if total_area > 0:

                        face_area_percent = (
                            face_area /
                            total_area
                        ) * 100

                    center_x = (
                        x1 + x2
                    ) / 2

                    center_y = (
                        y1 + y2
                    ) / 2

                    center_x_percent = (
                        center_x /
                        width
                    ) * 100

                    center_y_percent = (
                        center_y /
                        height
                    ) * 100

                    # ------------------------------------------------
                    # FACE BOX
                    # ------------------------------------------------

                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                # ------------------------------------------------
                # HEAD POSE
                # ------------------------------------------------

                pose = estimate_head_pose(
                    landmarks,
                    width,
                    height
                )

                if pose is not None:

                    yaw, pitch, roll = pose

                # ------------------------------------------------
                # DRAW LANDMARKS
                # ------------------------------------------------

                if draw_landmarks:

                    for index, landmark in enumerate(
                        landmarks
                    ):

                        px = int(
                            landmark.x *
                            width
                        )

                        py = int(
                            landmark.y *
                            height
                        )

                        if (
                            0 <= px < width
                            and
                            0 <= py < height
                        ):

                            # Draw every landmark
                            # as a small point.

                            cv2.circle(
                                display,
                                (px, py),
                                1,
                                (255, 255, 255),
                                -1
                            )

            # ====================================================
            # UPDATE STATISTICS
            # ====================================================

            current_stats = stats[
                current_test_id
            ]

            current_stats[
                "total_frames"
            ] += 1

            if face_detected:

                current_stats[
                    "face_detected_frames"
                ] += 1

                current_stats[
                    "landmark_success_frames"
                ] += 1

                current_stats[
                    "face_widths"
                ].append(
                    face_width
                )

                current_stats[
                    "face_heights"
                ].append(
                    face_height
                )

                current_stats[
                    "face_areas"
                ].append(
                    face_area_percent
                )

                if yaw is not None:

                    current_stats[
                        "yaws"
                    ].append(
                        yaw
                    )

                if pitch is not None:

                    current_stats[
                        "pitches"
                    ].append(
                        pitch
                    )

                if roll is not None:

                    current_stats[
                        "rolls"
                    ].append(
                        roll
                    )

            current_stats[
                "fps_values"
            ].append(
                fps
            )

            # ====================================================
            # DETECTION STATUS
            # ====================================================

            if face_detected:

                draw_text(
                    display,
                    "FACE DETECTED",
                    (20, 40),
                    0.75,
                    (0, 255, 0),
                    2
                )

            else:

                draw_text(
                    display,
                    "NO FACE DETECTED",
                    (20, 40),
                    0.75,
                    (0, 0, 255),
                    2
                )

            # ====================================================
            # TEST INFORMATION
            # ====================================================

            draw_text(
                display,
                (
                    f"TEST {current_test_id}: "
                    f"{current_test_name}"
                ),
                (20, 75),
                0.55,
                (0, 255, 255),
                2
            )

            # ====================================================
            # FACE INFORMATION
            # ====================================================

            draw_text(
                display,
                f"Landmarks: {landmark_count}",
                (20, 110),
                0.50,
                (255, 255, 255),
                1
            )

            draw_text(
                display,
                f"FPS: {fps:.1f}",
                (20, 140),
                0.50,
                (255, 255, 255),
                1
            )

            draw_text(
                display,
                f"Face width: {face_width}px",
                (20, 170),
                0.50,
                (255, 255, 255),
                1
            )

            draw_text(
                display,
                f"Face height: {face_height}px",
                (20, 200),
                0.50,
                (255, 255, 255),
                1
            )

            draw_text(
                display,
                (
                    f"Face area: "
                    f"{face_area_percent:.2f}%"
                ),
                (20, 230),
                0.50,
                (255, 255, 255),
                1
            )

            draw_text(
                display,
                (
                    f"Center X: "
                    f"{center_x_percent:.1f}%"
                ),
                (20, 260),
                0.50,
                (255, 255, 255),
                1
            )

            draw_text(
                display,
                (
                    f"Center Y: "
                    f"{center_y_percent:.1f}%"
                ),
                (20, 290),
                0.50,
                (255, 255, 255),
                1
            )

            # ====================================================
            # HEAD POSE
            # ====================================================

            if yaw is not None:

                draw_text(
                    display,
                    f"Yaw: {yaw:.1f} deg",
                    (20, 330),
                    0.52,
                    (255, 200, 0),
                    1
                )

                draw_text(
                    display,
                    f"Pitch: {pitch:.1f} deg",
                    (20, 360),
                    0.52,
                    (255, 200, 0),
                    1
                )

                draw_text(
                    display,
                    f"Roll: {roll:.1f} deg",
                    (20, 390),
                    0.52,
                    (255, 200, 0),
                    1
                )

            else:

                draw_text(
                    display,
                    "Head pose: unavailable",
                    (20, 330),
                    0.52,
                    (0, 165, 255),
                    1
                )

            # ====================================================
            # CURRENT DETECTION RATE
            # ====================================================

            current_total = current_stats[
                "total_frames"
            ]

            current_detected = current_stats[
                "face_detected_frames"
            ]

            if current_total > 0:

                current_rate = (
                    current_detected /
                    current_total
                ) * 100

            else:

                current_rate = 0.0

            draw_text(
                display,
                (
                    f"Test detection: "
                    f"{current_rate:.1f}%"
                ),
                (20, 430),
                0.52,
                (0, 255, 255),
                1
            )

            # ====================================================
            # TEST TIMER
            # ====================================================

            test_elapsed = (
                time.monotonic() -
                test_start_time
            )

            draw_text(
                display,
                (
                    f"Test time: "
                    f"{test_elapsed:.1f}s"
                ),
                (20, 460),
                0.50,
                (220, 220, 220),
                1
            )

            # ====================================================
            # INSTRUCTIONS
            # ====================================================

            instruction_y = (
                height - 145
            )

            draw_text(
                display,
                "1 Front | 2 Left | 3 Right",
                (
                    20,
                    instruction_y
                ),
                0.43,
                (220, 220, 220),
                1
            )

            draw_text(
                display,
                "4 Tilt-L | 5 Tilt-R | 6 Up | 7 Down",
                (
                    20,
                    instruction_y + 25
                ),
                0.43,
                (220, 220, 220),
                1
            )

            draw_text(
                display,
                "8 Near | 9 Far | D Landmarks",
                (
                    20,
                    instruction_y + 50
                ),
                0.43,
                (220, 220, 220),
                1
            )

            draw_text(
                display,
                "S Save test | Q Quit",
                (
                    20,
                    instruction_y + 75
                ),
                0.43,
                (220, 220, 220),
                1
            )

            # ====================================================
            # DRAW MODE
            # ====================================================

            draw_text(
                display,
                (
                    "LANDMARKS: ON"
                    if draw_landmarks
                    else
                    "LANDMARKS: OFF"
                ),
                (
                    width - 200,
                    35
                ),
                0.42,
                (
                    (0, 255, 0)
                    if draw_landmarks
                    else
                    (0, 165, 255)
                ),
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
            # SAVE FRAME DATA
            # ====================================================

            frame_writer.writerow([

                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                current_test_id,

                current_test_name,

                int(face_detected),

                landmark_count,

                face_width,

                face_height,

                round(
                    face_area_percent,
                    3
                ),

                round(
                    center_x_percent,
                    3
                ),

                round(
                    center_y_percent,
                    3
                ),

                round(
                    yaw,
                    3
                )
                if yaw is not None
                else
                "",

                round(
                    pitch,
                    3
                )
                if pitch is not None
                else
                "",

                round(
                    roll,
                    3
                )
                if roll is not None
                else
                "",

                round(
                    fps,
                    2
                )
            ])

            # Flush occasionally
            if (
                current_stats["total_frames"]
                % 30
                == 0
            ):

                frame_file.flush()

            # ====================================================
            # KEYBOARD
            # ====================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ----------------------------------------------------
            # QUIT
            # ----------------------------------------------------

            if key == ord("q"):

                break

            # ----------------------------------------------------
            # LANDMARK TOGGLE
            # ----------------------------------------------------

            elif key == ord("d"):

                draw_landmarks = (
                    not draw_landmarks
                )

                print(
                    "Landmarks:",
                    "ON"
                    if draw_landmarks
                    else
                    "OFF"
                )

            # ----------------------------------------------------
            # SAVE CURRENT TEST
            # ----------------------------------------------------

            elif key == ord("s"):

                save_test_summary(
                    current_test_id
                )

            # ----------------------------------------------------
            # TEST 1
            # ----------------------------------------------------

            elif key == ord("1"):

                save_test_summary(
                    current_test_id
                )

                reset_test("1")

            # ----------------------------------------------------
            # TEST 2
            # ----------------------------------------------------

            elif key == ord("2"):

                save_test_summary(
                    current_test_id
                )

                reset_test("2")

            # ----------------------------------------------------
            # TEST 3
            # ----------------------------------------------------

            elif key == ord("3"):

                save_test_summary(
                    current_test_id
                )

                reset_test("3")

            # ----------------------------------------------------
            # TEST 4
            # ----------------------------------------------------

            elif key == ord("4"):

                save_test_summary(
                    current_test_id
                )

                reset_test("4")

            # ----------------------------------------------------
            # TEST 5
            # ----------------------------------------------------

            elif key == ord("5"):

                save_test_summary(
                    current_test_id
                )

                reset_test("5")

            # ----------------------------------------------------
            # TEST 6
            # ----------------------------------------------------

            elif key == ord("6"):

                save_test_summary(
                    current_test_id
                )

                reset_test("6")

            # ----------------------------------------------------
            # TEST 7
            # ----------------------------------------------------

            elif key == ord("7"):

                save_test_summary(
                    current_test_id
                )

                reset_test("7")

            # ----------------------------------------------------
            # TEST 8
            # ----------------------------------------------------

            elif key == ord("8"):

                save_test_summary(
                    current_test_id
                )

                reset_test("8")

            # ----------------------------------------------------
            # TEST 9
            # ----------------------------------------------------

            elif key == ord("9"):

                save_test_summary(
                    current_test_id
                )

                reset_test("9")


finally:

    # ============================================================
    # SAVE FINAL TEST
    # ============================================================

    save_test_summary(
        current_test_id
    )

    # ============================================================
    # RELEASE RESOURCES
    # ============================================================

    camera.release()

    cv2.destroyAllWindows()

    frame_file.flush()

    summary_file.flush()

    frame_file.close()

    summary_file.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("FEASIBILITY TEST COMPLETE")
print("=" * 70)
print()
print(
    "Frame data:"
)
print(
    FRAME_CSV
)
print()
print(
    "Test summaries:"
)
print(
    SUMMARY_CSV
)
print()
print(
    "Your research data is now saved."
)
print()
print("=" * 70)