from pathlib import Path
import time

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"


# MediaPipe Face Mesh landmark indices for the eye regions.
# These are used only for visualising the eye areas at this stage.
LEFT_EYE = [
    362, 385, 387, 263, 373, 380,
    466, 388, 386, 374, 398
]

RIGHT_EYE = [
    33, 160, 158, 133, 153, 144,
    246, 161, 159, 145, 173
]


def draw_eye_landmarks(frame, face_landmarks, indices, width, height):
    for index in indices:
        landmark = face_landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 255),
                -1
            )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError("Unable to open camera.")

    print("Eye landmark test started.")
    print("Press Q to exit.")

    start_time = time.perf_counter()

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Unable to read camera frame.")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                height, width, _ = frame.shape

                if result.face_landmarks:

                    face = result.face_landmarks[0]

                    draw_eye_landmarks(
                        frame,
                        face,
                        LEFT_EYE,
                        width,
                        height
                    )

                    draw_eye_landmarks(
                        frame,
                        face,
                        RIGHT_EYE,
                        width,
                        height
                    )

                    status = "EYE LANDMARKS DETECTED"

                else:
                    status = "NO FACE DETECTED"

                cv2.putText(
                    frame,
                    status,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Press Q to exit",
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Sleeping Eye Detection - Eye Landmarks",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print("Eye landmark test stopped.")


if __name__ == "__main__":
    main()