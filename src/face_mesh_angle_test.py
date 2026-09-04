import cv2
import mediapipe as mp
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# SETTINGS
# ============================================================

FACE_LANDMARKER_MODEL = (
    r"D:\SleepingEyeDetection\face_landmarker.task"
)

CAMERA_INDEX = 0


# ============================================================
# FACE MESH / LANDMARKER
# ============================================================

def main():

    print("=" * 60)
    print("FACE MESH ANGLE TEST")
    print("=" * 60)

    print("Loading MediaPipe Face Landmarker...")

    base_options = python.BaseOptions(
        model_asset_path=FACE_LANDMARKER_MODEL
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True
    )

    landmarker = vision.FaceLandmarker.create_from_options(
        options
    )

    print("MediaPipe Face Landmarker loaded.")
    print()

    # ========================================================
    # CAMERA
    # ========================================================

    print("Opening camera...")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print("ERROR: Camera could not be opened.")

        landmarker.close()

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print("Camera started.")
    print()
    print("ANGLE TEST")
    print("------------------------------")
    print("Look straight")
    print("Slowly turn LEFT")
    print("Slowly turn RIGHT")
    print("Look UP")
    print("Look DOWN")
    print()
    print("Press Q to quit.")
    print("=" * 60)

    start_time = time.time()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Could not read camera.")

            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        h, w = frame.shape[:2]

        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        # ----------------------------------------------------
        # Detect face
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        face_count = len(
            result.face_landmarks
        )

        # ----------------------------------------------------
        # No face
        # ----------------------------------------------------

        if face_count == 0:

            cv2.putText(
                frame,
                "NO FACE",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        else:

            landmarks = result.face_landmarks[0]

            # ------------------------------------------------
            # Draw complete face mesh
            # ------------------------------------------------

            for landmark in landmarks:

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )

                if (
                    0 <= x < w
                    and
                    0 <= y < h
                ):

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            # ------------------------------------------------
            # Find approximate eye regions
            # ------------------------------------------------

            # MediaPipe Face Mesh eye landmark indices
            left_eye_indices = [
                33, 133, 160, 159, 158,
                157, 173, 144, 145, 153
            ]

            right_eye_indices = [
                362, 263, 387, 386, 385,
                384, 398, 373, 374, 380
            ]

            def draw_eye(
                indices,
                label
            ):

                points = []

                for index in indices:

                    if index >= len(landmarks):
                        continue

                    lm = landmarks[index]

                    x = int(
                        lm.x * w
                    )

                    y = int(
                        lm.y * h
                    )

                    points.append(
                        (x, y)
                    )

                if len(points) < 2:
                    return False

                xs = [
                    p[0]
                    for p in points
                ]

                ys = [
                    p[1]
                    for p in points
                ]

                x1 = max(
                    0,
                    min(xs) - 10
                )

                y1 = max(
                    0,
                    min(ys) - 10
                )

                x2 = min(
                    w - 1,
                    max(xs) + 10
                )

                y2 = min(
                    h - 1,
                    max(ys) + 10
                )

                # Eye box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    2
                )

                # Eye center
                cx = int(
                    sum(xs) / len(xs)
                )

                cy = int(
                    sum(ys) / len(ys)
                )

                cv2.circle(
                    frame,
                    (cx, cy),
                    4,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )

                return True

            left_visible = draw_eye(
                left_eye_indices,
                "LEFT EYE"
            )

            right_visible = draw_eye(
                right_eye_indices,
                "RIGHT EYE"
            )

            # ------------------------------------------------
            # Visibility status
            # ------------------------------------------------

            if left_visible:
                left_status = "VISIBLE"
            else:
                left_status = "NOT VISIBLE"

            if right_visible:
                right_status = "VISIBLE"
            else:
                right_status = "NOT VISIBLE"

            # ------------------------------------------------
            # Information
            # ------------------------------------------------

            cv2.putText(
                frame,
                "FACE MESH ACTIVE",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Faces detected: {face_count}",
                (30, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Left eye: {left_status}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right eye: {right_status}",
                (30, 142),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Face Mesh Angle Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    landmarker.close()

    cv2.destroyAllWindows()

    print()
    print("Camera stopped.")
    print("Face mesh angle test completed.")


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()