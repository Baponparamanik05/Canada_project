import cv2
import mediapipe as mp
import os
import math
import time

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\SleepingEyeDetection"
MODEL_PATH = os.path.join(PROJECT_DIR, "face_landmarker.task")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "eye_dataset")

IMG_SIZE = 224

# MediaPipe Face Landmarker eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Extra points used for rough head-angle estimation
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_FACE = 234
RIGHT_FACE = 454


# ============================================================
# CREATE DATASET FOLDERS
# ============================================================

for eye in ["left", "right"]:
    for state in ["open", "closed"]:
        folder = os.path.join(OUTPUT_DIR, eye, state)
        os.makedirs(folder, exist_ok=True)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def landmark_xy(landmarks, index, width, height):
    p = landmarks[index]

    x = int(p.x * width)
    y = int(p.y * height)

    return x, y


def eye_box(landmarks, indices, width, height, padding=0.35):
    points = []

    for idx in indices:
        x, y = landmark_xy(landmarks, idx, width, height)
        points.append((x, y))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    w = max_x - min_x
    h = max_y - min_y

    # Add padding around eye
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, min_x - pad_x)
    y1 = max(0, min_y - pad_y)

    x2 = min(width, max_x + pad_x)
    y2 = min(height, max_y + pad_y)

    return x1, y1, x2, y2


def calculate_eye_ear(landmarks, indices):
    """
    Approximate EAR using the six eye landmarks.
    """

    p1 = landmarks[indices[0]]
    p2 = landmarks[indices[1]]
    p3 = landmarks[indices[2]]
    p4 = landmarks[indices[3]]
    p5 = landmarks[indices[4]]
    p6 = landmarks[indices[5]]

    def dist(a, b):
        return math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2
        )

    horizontal = dist(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    vertical1 = dist(p2, p6)
    vertical2 = dist(p3, p5)

    return (vertical1 + vertical2) / (2.0 * horizontal)


def estimate_angles(landmarks):
    """
    Rough head pose estimate.

    This is intended for dataset filenames only.
    It is NOT a replacement for 3DDFA head pose.
    """

    nose = landmarks[NOSE_TIP]
    left = landmarks[LEFT_FACE]
    right = landmarks[RIGHT_FACE]
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]

    # Horizontal position of nose relative to face width
    face_width = right.x - left.x

    if abs(face_width) < 1e-6:
        yaw = 0
    else:
        face_center = (left.x + right.x) / 2.0
        yaw_ratio = (nose.x - face_center) / face_width
        yaw = yaw_ratio * 90

    # Vertical position
    face_height = chin.y - forehead.y

    if abs(face_height) < 1e-6:
        pitch = 0
    else:
        face_center_y = (forehead.y + chin.y) / 2.0
        pitch_ratio = (nose.y - face_center_y) / face_height
        pitch = pitch_ratio * 90

    yaw = max(-60, min(60, yaw))
    pitch = max(-60, min(60, pitch))

    return int(round(yaw)), int(round(pitch))


def crop_eye(frame, landmarks, indices):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = eye_box(
        landmarks,
        indices,
        w,
        h
    )

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, (x1, y1, x2, y2)

    crop = cv2.resize(
        crop,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return crop, (x1, y1, x2, y2)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("SLEEPING EYE DETECTION - EYE IMAGE COLLECTOR")
    print("=" * 65)
    print()
    print("Controls:")
    print("  O = save OPEN eye images")
    print("  C = save CLOSED eye images")
    print("  SPACE = pause/resume")
    print("  Q = quit")
    print()
    print("Images will be saved as 224 x 224 pixels.")
    print()
    print("Dataset location:")
    print(OUTPUT_DIR)
    print("=" * 65)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_timestamp = 0

    save_mode = None
    paused = False

    image_counter = 0

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            if not paused:
                ret, frame = cap.read()

                if not ret:
                    print("ERROR: Could not read camera.")
                    break

                frame_timestamp += 33

            display = frame.copy()

            h, w = display.shape[:2]

            # ------------------------------------------------
            # MEDIAPIPE IMAGE
            # ------------------------------------------------

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect_for_video(
                mp_image,
                frame_timestamp
            )

            faces = result.face_landmarks

            # ------------------------------------------------
            # HEADER
            # ------------------------------------------------

            cv2.putText(
                display,
                "EYE IMAGE COLLECTOR",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                "O=OPEN   C=CLOSED   SPACE=PAUSE   Q=QUIT",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # MODE
            # ------------------------------------------------

            if save_mode == "open":
                mode_text = "SAVE MODE: OPEN"
                mode_color = (0, 255, 0)

            elif save_mode == "closed":
                mode_text = "SAVE MODE: CLOSED"
                mode_color = (0, 165, 255)

            else:
                mode_text = "SAVE MODE: OFF"
                mode_color = (255, 255, 255)

            cv2.putText(
                display,
                mode_text,
                (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                mode_color,
                2
            )

            # ------------------------------------------------
            # FACE DETECTED
            # ------------------------------------------------

            if len(faces) > 0:

                landmarks = faces[0]

                # --------------------------------------------
                # FACE MESH
                # --------------------------------------------

                for p in landmarks:

                    px = int(p.x * w)
                    py = int(p.y * h)

                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(
                            display,
                            (px, py),
                            1,
                            (0, 255, 0),
                            -1
                        )

                # --------------------------------------------
                # EYE CROPS
                # --------------------------------------------

                left_crop, left_box = crop_eye(
                    frame,
                    landmarks,
                    LEFT_EYE
                )

                right_crop, right_box = crop_eye(
                    frame,
                    landmarks,
                    RIGHT_EYE
                )

                # --------------------------------------------
                # DRAW LEFT EYE
                # --------------------------------------------

                lx1, ly1, lx2, ly2 = left_box

                cv2.rectangle(
                    display,
                    (lx1, ly1),
                    (lx2, ly2),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    "LEFT EYE",
                    (lx1, max(ly1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                # --------------------------------------------
                # DRAW RIGHT EYE
                # --------------------------------------------

                rx1, ry1, rx2, ry2 = right_box

                cv2.rectangle(
                    display,
                    (rx1, ry1),
                    (rx2, ry2),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    "RIGHT EYE",
                    (rx1, max(ry1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                # --------------------------------------------
                # EAR
                # --------------------------------------------

                left_ear = calculate_eye_ear(
                    landmarks,
                    LEFT_EYE
                )

                right_ear = calculate_eye_ear(
                    landmarks,
                    RIGHT_EYE
                )

                # --------------------------------------------
                # ANGLES
                # --------------------------------------------

                yaw, pitch = estimate_angles(landmarks)

                cv2.putText(
                    display,
                    f"Approx Yaw: {yaw} deg",
                    (25, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"Approx Pitch: {pitch} deg",
                    (25, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"L EAR: {left_ear:.3f}",
                    (w - 250, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"R EAR: {right_ear:.3f}",
                    (w - 250, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # --------------------------------------------
                # SAVE IMAGES
                # --------------------------------------------

                if save_mode is not None:

                    timestamp = int(time.time() * 1000)

                    state = save_mode

                    if left_crop is not None:

                        filename = (
                            f"left_{state}"
                            f"_yaw{yaw:+03d}"
                            f"_pitch{pitch:+03d}"
                            f"_{timestamp}.jpg"
                        )

                        path = os.path.join(
                            OUTPUT_DIR,
                            "left",
                            state,
                            filename
                        )

                        cv2.imwrite(
                            path,
                            left_crop
                        )

                        image_counter += 1

                    if right_crop is not None:

                        filename = (
                            f"right_{state}"
                            f"_yaw{yaw:+03d}"
                            f"_pitch{pitch:+03d}"
                            f"_{timestamp}.jpg"
                        )

                        path = os.path.join(
                            OUTPUT_DIR,
                            "right",
                            state,
                            filename
                        )

                        cv2.imwrite(
                            path,
                            right_crop
                        )

                        image_counter += 1

                    # Prevent saving thousands of images per second
                    time.sleep(0.08)

            else:

                cv2.putText(
                    display,
                    "NO FACE DETECTED",
                    (25, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # ------------------------------------------------
            # IMAGE COUNT
            # ------------------------------------------------

            cv2.putText(
                display,
                f"Images saved: {image_counter}",
                (25, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            if paused:
                cv2.putText(
                    display,
                    "PAUSED",
                    (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

            cv2.imshow(
                "Eye Image Collector",
                display
            )

            # ------------------------------------------------
            # KEYBOARD
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("o"):
                save_mode = "open"
                print("SAVE MODE -> OPEN")

            elif key == ord("c"):
                save_mode = "closed"
                print("SAVE MODE -> CLOSED")

            elif key == ord(" "):
                paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 65)
    print("COLLECTION FINISHED")
    print("=" * 65)
    print(f"Total images saved: {image_counter}")
    print(f"Dataset location: {OUTPUT_DIR}")
    print("=" * 65)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()