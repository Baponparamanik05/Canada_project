import cv2
import mediapipe as mp
import time
import math
from pathlib import Path
from collections import deque
import threading
import winsound

from drowsiness_engine import DrowsinessEngine


# ============================================================
# SLEEPING EYE DETECTION
# FAST + STABLE PER-EYE EAR DETECTION
#
# IMPORTANT:
# - EAR and MediaPipe eyeBlink blendshapes are combined.
# - Strong blink scores can rescue EAR when eyelid landmarks
#   become unreliable (low light / head tilt / occlusion).
# - LEFT/RIGHT are anatomical eyes:
#       YOUR LEFT  = screen RIGHT in mirror view
#       YOUR RIGHT = screen LEFT in mirror view
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

CAMERA_INDEX = 0
# 640x480 is substantially faster on a CPU while still giving
# enough resolution for face/eye landmarks.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ============================================================
# SPEED / ACCURACY SETTINGS
# ============================================================

# Keep this short so OPEN/CLOSED changes appear quickly.
SMOOTHING_WINDOW = 1

# 2 frames gives fast response while rejecting one bad frame.
STATE_CONFIRMATION_FRAMES = 1

# Calibration: keep BOTH eyes naturally open.
CALIBRATION_SECONDS = 3.0
MIN_CALIBRATION_SAMPLES = 30

# Eye state uses a ratio against each eye's OWN open baseline.
#
# Example:
#   baseline = 0.20
#   close threshold = 0.20 * 0.62 = 0.124
#   open threshold  = 0.20 * 0.82 = 0.164
#
# Between close/open thresholds, keep the previous state.
CLOSE_RATIO = 0.62
OPEN_RATIO = 0.82

# MediaPipe Face Landmarker blendshape scores:
# eyeBlinkLeft / eyeBlinkRight are approximately:
#   0.0 = open
#   1.0 = closed
#
# They are used as a SECOND signal when EAR is unreliable,
# especially with low light, head tilt, or partially occluded eyes.
BLINK_CLOSED_SCORE = 0.50
BLINK_OPEN_SCORE = 0.20

# Safety limits for very unusual camera conditions.
MIN_CLOSE_THRESHOLD = 0.075
MAX_CLOSE_THRESHOLD = 0.150

# Slowly adapt the open baseline only when an eye is confidently open.
# This handles small head-position changes without learning a closed eye.
BASELINE_ADAPT_ALPHA = 0.015

# Drowsiness timing.
DROWSY_SECONDS = 0.8
SLEEPING_SECONDS = 2.0
RECOVERY_SECONDS = 0.25


# ============================================================
# UI
# ============================================================

BLACK = (18, 18, 18)
DARK_PANEL = (28, 28, 28)
BORDER = (65, 65, 65)

WHITE = (245, 245, 245)
MUTED = (155, 155, 155)

GREEN = (70, 220, 70)
RED = (70, 70, 235)
YELLOW = (40, 200, 245)
ORANGE = (40, 165, 255)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_vision = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions


# ============================================================
# ANATOMICAL EYE LANDMARKS
# ============================================================
#
# MediaPipe Face Mesh:
#
# 362...263 = anatomical LEFT eye
# 33...133  = anatomical RIGHT eye
#
# Because the camera is mirrored:
# YOUR LEFT  -> screen RIGHT
# YOUR RIGHT -> screen LEFT
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ============================================================
# HELPERS
# ============================================================

def distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1],
    )


def calculate_ear(face, indices, width, height):
    """
    Eye Aspect Ratio.

    Higher EAR = more open.
    Lower EAR  = more closed.
    """

    points = []

    for idx in indices:
        lm = face[idx]
        points.append(
            (
                lm.x * width,
                lm.y * height,
            )
        )

    p1, p2, p3, p4, p5, p6 = points

    horizontal = distance(p1, p4)

    if horizontal <= 1e-6:
        return 0.0

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    return (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )


def get_blendshape_score(result, name):
    """Return one MediaPipe face-blendshape score by name."""
    try:
        if not result.face_blendshapes:
            return 0.0

        categories = result.face_blendshapes[0]

        for category in categories:
            if category.category_name == name:
                return float(category.score)

    except Exception:
        pass

    return 0.0


def classify_eye_combined(
    ear,
    blendshape,
    current_state,
    close_threshold,
    open_threshold,
):
    """
    Fast per-eye classifier using BOTH signals.

    Priority:
      1. Strong MediaPipe blink score.
      2. Otherwise EAR.
      3. Hysteresis in the uncertain zone.

    This prevents an unreliable EAR measurement from keeping
    an obviously closed eye marked OPEN.
    """

    # Strong blendshape evidence.
    if blendshape >= BLINK_CLOSED_SCORE:
        return "CLOSED"

    if blendshape <= BLINK_OPEN_SCORE:
        if ear <= close_threshold:
            return "CLOSED"
        return "OPEN"

    # Blendshape is uncertain -> use EAR.
    if ear <= close_threshold:
        return "CLOSED"

    if ear >= open_threshold:
        return "OPEN"

    if current_state in ("OPEN", "CLOSED"):
        return current_state

    return "OPEN"


def get_eye_box(face, indices, width, height, pad=10):
    xs = [int(face[i].x * width) for i in indices]
    ys = [int(face[i].y * height) for i in indices]

    if not xs:
        return None

    return (
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(width - 1, max(xs) + pad),
        min(height - 1, max(ys) + pad),
    )


def median(values):
    if not values:
        return 0.0

    values = sorted(values)
    n = len(values)
    middle = n // 2

    if n % 2:
        return values[middle]

    return (
        values[middle - 1]
        + values[middle]
    ) / 2.0


def draw_text(
    frame,
    text,
    xy,
    size=0.5,
    color=WHITE,
    thickness=1,
):
    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(frame, x1, y1, x2, y2):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        DARK_PANEL,
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.94,
        frame,
        0.06,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        BORDER,
        1,
    )


def draw_dot(frame, x, y, active=True):
    cv2.circle(
        frame,
        (x, y),
        6,
        GREEN if active else RED,
        -1,
    )


def stable_state_update(
    current_state,
    candidate,
    candidate_frames,
    new_state,
):
    """
    FAST + STABLE state update.

    A clearly CLOSED or clearly OPEN eye changes immediately.
    The uncertain EAR zone keeps the previous state.

    This is intentionally faster than waiting for two frames:
    if the measured EAR is already far below the close threshold,
    we should not display OPEN while the eye is visibly closed.
    """

    # Clear state = immediate change.
    if new_state in ("OPEN", "CLOSED"):
        current_state = new_state
        candidate = new_state
        candidate_frames = STATE_CONFIRMATION_FRAMES

    else:
        # Only use the candidate counter for an uncertain state.
        candidate = new_state
        candidate_frames = 1

    return (
        current_state,
        candidate,
        candidate_frames,
    )


def classify_eye_ear(
    ear,
    current_state,
    baseline,
    close_threshold,
    open_threshold,
):
    """
    Per-eye hysteresis.

    CLOSED:
        EAR <= close threshold

    OPEN:
        EAR >= open threshold

    In between:
        keep previous state

    This is important because it prevents flickering.
    """

    if ear <= close_threshold:
        return "CLOSED"

    if ear >= open_threshold:
        return "OPEN"

    # Do not force a new state in the uncertain zone.
    if current_state in ("OPEN", "CLOSED"):
        return current_state

    return "OPEN"


def build_thresholds(baseline):
    close_threshold = baseline * CLOSE_RATIO
    open_threshold = baseline * OPEN_RATIO

    close_threshold = max(
        MIN_CLOSE_THRESHOLD,
        min(MAX_CLOSE_THRESHOLD, close_threshold),
    )

    # Always keep OPEN threshold above CLOSE threshold.
    open_threshold = max(
        open_threshold,
        close_threshold + 0.025,
    )

    return close_threshold, open_threshold


# ============================================================
# ALARM
# ============================================================

class AlarmController:

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self.active = False

    def _loop(self):
        while not self._stop_event.is_set():

            try:
                winsound.Beep(1400, 220)

                if self._stop_event.wait(0.08):
                    break

                winsound.Beep(1900, 220)

                if self._stop_event.wait(0.08):
                    break

            except Exception:

                try:
                    winsound.MessageBeep(
                        winsound.MB_ICONEXCLAMATION
                    )
                except Exception:
                    pass

    def start(self):

        if self.active:
            return

        self._stop_event.clear()
        self.active = True

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )

        self._thread.start()

    def stop(self):

        if not self.active:
            return

        self._stop_event.set()
        self.active = False

        if self._thread is not None:
            self._thread.join(timeout=0.5)

        self._thread = None

    def close(self):
        self.stop()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("SLEEPING EYE DETECTION")
    print("FAST + STABLE EAR + BLENDSHAPE VERSION")
    print("Resolution: 640x480 | Eye decision: EAR + eyeBlink")
    print("=" * 65)

    if not MODEL_PATH.exists():

        print("\nERROR: Model not found:")
        print(MODEL_PATH)

        return

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():

        print("\nERROR: Could not open camera.")
        print("Try CAMERA_INDEX = 1.")

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    options = mp_vision.FaceLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),

        running_mode=mp_vision.RunningMode.VIDEO,

        num_faces=1,

        min_face_detection_confidence=0.55,
        min_face_presence_confidence=0.55,
        min_tracking_confidence=0.55,

        # We keep blendshapes available for future work,
        # but eye state is calculated from EAR only.
        output_face_blendshapes=True,
    )

    face_landmarker = (
        mp_vision.FaceLandmarker.create_from_options(
            options
        )
    )

    left_history = deque(
        maxlen=SMOOTHING_WINDOW
    )

    right_history = deque(
        maxlen=SMOOTHING_WINDOW
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    calibration_start = time.monotonic()

    left_calibration = []
    right_calibration = []

    calibrated = False

    left_baseline = 0.20
    right_baseline = 0.20

    left_close = 0.12
    right_close = 0.12

    left_open = 0.16
    right_open = 0.16

    left_state = "CALIBRATING"
    right_state = "CALIBRATING"

    left_candidate = "CALIBRATING"
    right_candidate = "CALIBRATING"

    left_candidate_frames = 0
    right_candidate_frames = 0

    # --------------------------------------------------------
    # DROWSINESS
    # --------------------------------------------------------

    engine = DrowsinessEngine(
        drowsy_seconds=DROWSY_SECONDS,
        sleeping_seconds=SLEEPING_SECONDS,
        recovery_seconds=RECOVERY_SECONDS,
    )

    alarm = AlarmController()

    drowsiness_state = "NORMAL"
    closed_duration = 0.0
    blink_count = 0

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = 0.0
    fps_frames = 0
    fps_start = time.monotonic()

    print("\nCALIBRATION")
    print("-" * 50)
    print("Keep BOTH eyes naturally OPEN.")
    print("Look normally at the camera for 3 seconds.")
    print("Do not blink intentionally during calibration.")
    print("After calibration, test:")
    print("  1. Both eyes open")
    print("  2. Close LEFT eye only")
    print("  3. Open LEFT eye")
    print("  4. Close RIGHT eye only")
    print("  5. Open RIGHT eye")
    print("  6. Close BOTH eyes")
    print("\nPress R to recalibrate.")
    print("Press Q to exit.\n")

    try:

        while True:

            ok, frame = camera.read()

            if not ok:

                print(
                    "ERROR: Could not read camera frame."
                )

                break

            # Mirror view.
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            fps_frames += 1

            fps_elapsed = (
                time.monotonic()
                - fps_start
            )

            if fps_elapsed >= 1.0:

                fps = (
                    fps_frames
                    / fps_elapsed
                )

                fps_frames = 0
                fps_start = time.monotonic()

            # ------------------------------------------------
            # MEDIAPIPE
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int(
                time.monotonic() * 1000
            )

            result = (
                face_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

            face_detected = (
                len(result.face_landmarks) > 0
            )

            # ------------------------------------------------
            # HEADER
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (0, 0),
                (width, 92),
                BLACK,
                -1,
            )

            draw_text(
                frame,
                "SLEEPING EYE DETECTION",
                (28, 38),
                0.85,
                WHITE,
                2,
            )

            draw_text(
                frame,
                "EAR ONLY  |  FAST RESPONSE  |  MIRROR VIEW",
                (30, 68),
                0.40,
                MUTED,
                1,
            )

            # ------------------------------------------------
            # RIGHT PANEL
            # ------------------------------------------------

            panel_width = 360

            panel_x1 = (
                width
                - panel_width
                - 25
            )

            panel_y1 = 110

            panel_x2 = width - 25
            panel_y2 = height - 25

            draw_panel(
                frame,
                panel_x1,
                panel_y1,
                panel_x2,
                panel_y2,
            )

            # ------------------------------------------------
            # SYSTEM STATUS
            # ------------------------------------------------

            draw_text(
                frame,
                "SYSTEM STATUS",
                (
                    panel_x1 + 20,
                    panel_y1 + 32,
                ),
                0.55,
                WHITE,
                2,
            )

            draw_dot(
                frame,
                panel_x1 + 28,
                panel_y1 + 65,
                True,
            )

            draw_text(
                frame,
                "Camera",
                (
                    panel_x1 + 45,
                    panel_y1 + 70,
                ),
                0.48,
                WHITE,
            )

            draw_text(
                frame,
                "READY",
                (
                    panel_x2 - 78,
                    panel_y1 + 70,
                ),
                0.43,
                GREEN,
            )

            draw_dot(
                frame,
                panel_x1 + 28,
                panel_y1 + 100,
                face_detected,
            )

            draw_text(
                frame,
                "Face Tracking",
                (
                    panel_x1 + 45,
                    panel_y1 + 105,
                ),
                0.48,
                WHITE,
            )

            draw_text(
                frame,
                (
                    "ACTIVE"
                    if face_detected
                    else "WAITING"
                ),
                (
                    panel_x2 - 90,
                    panel_y1 + 105,
                ),
                0.43,
                (
                    GREEN
                    if face_detected
                    else MUTED
                ),
            )

            cv2.line(
                frame,
                (
                    panel_x1 + 20,
                    panel_y1 + 130,
                ),
                (
                    panel_x2 - 20,
                    panel_y1 + 130,
                ),
                BORDER,
                1,
            )

            # ------------------------------------------------
            # VALUES
            # ------------------------------------------------

            left_ear = 0.0
            right_ear = 0.0
            left_blink = 0.0
            right_blink = 0.0

            if face_detected:

                face = result.face_landmarks[0]

                # EAR for anatomical LEFT eye.
                left_raw = calculate_ear(
                    face,
                    LEFT_EYE,
                    width,
                    height,
                )

                # EAR for anatomical RIGHT eye.
                right_raw = calculate_ear(
                    face,
                    RIGHT_EYE,
                    width,
                    height,
                )

                left_history.append(
                    left_raw
                )

                right_history.append(
                    right_raw
                )

                left_ear = (
                    sum(left_history)
                    / len(left_history)
                )

                right_ear = (
                    sum(right_history)
                    / len(right_history)
                )

                # =================================================
                # CALIBRATION
                # =================================================

                if not calibrated:

                    elapsed = (
                        time.monotonic()
                        - calibration_start
                    )

                    # During calibration, only accept samples
                    # that look like normally open eyes.
                    calibration_left_blink = get_blendshape_score(
                        result,
                        "eyeBlinkLeft",
                    )
                    calibration_right_blink = get_blendshape_score(
                        result,
                        "eyeBlinkRight",
                    )

                    if (
                        left_ear > 0.14
                        and right_ear > 0.14
                        and calibration_left_blink < BLINK_OPEN_SCORE
                        and calibration_right_blink < BLINK_OPEN_SCORE
                    ):

                        left_calibration.append(
                            left_ear
                        )

                        right_calibration.append(
                            right_ear
                        )

                    remaining = max(
                        0.0,
                        CALIBRATION_SECONDS
                        - elapsed,
                    )

                    left_state = "CALIBRATING"
                    right_state = "CALIBRATING"

                    enough_samples = (
                        len(left_calibration)
                        >= MIN_CALIBRATION_SAMPLES
                        and
                        len(right_calibration)
                        >= MIN_CALIBRATION_SAMPLES
                    )

                    if (
                        elapsed >= CALIBRATION_SECONDS
                        and enough_samples
                    ):

                        left_baseline = median(
                            left_calibration
                        )

                        right_baseline = median(
                            right_calibration
                        )

                        (
                            left_close,
                            left_open,
                        ) = build_thresholds(
                            left_baseline
                        )

                        (
                            right_close,
                            right_open,
                        ) = build_thresholds(
                            right_baseline
                        )

                        calibrated = True

                        left_state = "OPEN"
                        right_state = "OPEN"

                        left_candidate = "OPEN"
                        right_candidate = "OPEN"

                        left_candidate_frames = (
                            STATE_CONFIRMATION_FRAMES
                        )

                        right_candidate_frames = (
                            STATE_CONFIRMATION_FRAMES
                        )

                        print("\n")
                        print("=" * 60)
                        print("CALIBRATION COMPLETE")
                        print("=" * 60)
                        print(
                            f"LEFT baseline  : {left_baseline:.3f}"
                        )
                        print(
                            f"LEFT close     : {left_close:.3f}"
                        )
                        print(
                            f"LEFT open      : {left_open:.3f}"
                        )
                        print(
                            f"RIGHT baseline : {right_baseline:.3f}"
                        )
                        print(
                            f"RIGHT close    : {right_close:.3f}"
                        )
                        print(
                            f"RIGHT open     : {right_open:.3f}"
                        )
                        print("=" * 60)
                        print()

                # =================================================
                # CLASSIFICATION
                # =================================================

                else:

                    # EAR is the ONLY signal controlling eye state.
                    #
                    # This removes the previous false CLOSED result
                    # caused by blendshape scores.

                    # MediaPipe blink scores provide a second,
                    # independent signal when EAR is unreliable.
                    left_blink = get_blendshape_score(
                        result,
                        "eyeBlinkLeft",
                    )

                    right_blink = get_blendshape_score(
                        result,
                        "eyeBlinkRight",
                    )

                    left_new = classify_eye_combined(
                        left_ear,
                        left_blink,
                        left_state,
                        left_close,
                        left_open,
                    )

                    right_new = classify_eye_combined(
                        right_ear,
                        right_blink,
                        right_state,
                        right_close,
                        right_open,
                    )

                    (
                        left_state,
                        left_candidate,
                        left_candidate_frames,
                    ) = stable_state_update(
                        left_state,
                        left_candidate,
                        left_candidate_frames,
                        left_new,
                    )

                    (
                        right_state,
                        right_candidate,
                        right_candidate_frames,
                    ) = stable_state_update(
                        right_state,
                        right_candidate,
                        right_candidate_frames,
                        right_new,
                    )

                    # ------------------------------------------------
                    # SLOW BASELINE ADAPTATION
                    # ONLY when the eye is confidently OPEN.
                    # ------------------------------------------------

                    if (
                        left_state == "OPEN"
                        and left_ear > left_open
                    ):

                        left_baseline = (
                            (1.0 - BASELINE_ADAPT_ALPHA)
                            * left_baseline
                            +
                            BASELINE_ADAPT_ALPHA
                            * left_ear
                        )

                        (
                            left_close,
                            left_open,
                        ) = build_thresholds(
                            left_baseline
                        )

                    if (
                        right_state == "OPEN"
                        and right_ear > right_open
                    ):

                        right_baseline = (
                            (1.0 - BASELINE_ADAPT_ALPHA)
                            * right_baseline
                            +
                            BASELINE_ADAPT_ALPHA
                            * right_ear
                        )

                        (
                            right_close,
                            right_open,
                        ) = build_thresholds(
                            right_baseline
                        )

                    # ------------------------------------------------
                    # DROWSINESS
                    # ------------------------------------------------
                    #
                    # IMPORTANT:
                    # One closed eye DOES NOT mean sleeping.
                    #
                    # Both anatomical eyes must be CLOSED.
                    # ------------------------------------------------

                    both_closed = (
                        left_state == "CLOSED"
                        and
                        right_state == "CLOSED"
                    ) or (
                        left_blink >= BLINK_CLOSED_SCORE
                        and
                        right_blink >= BLINK_CLOSED_SCORE
                    )

                    status = engine.update(
                        both_closed
                    )

                    drowsiness_state = (
                        status["state"]
                    )

                    closed_duration = (
                        status["closed_duration"]
                    )

                    blink_count = (
                        status["blink_count"]
                    )

                    # Alarm only when BOTH eyes have remained
                    # closed long enough to reach SLEEPING.
                    if (
                        drowsiness_state
                        == "SLEEPING"
                    ):
                        alarm.start()
                    else:
                        alarm.stop()

                # ------------------------------------------------
                # EYE BOXES
                # ------------------------------------------------

                left_box = get_eye_box(
                    face,
                    LEFT_EYE,
                    width,
                    height,
                )

                right_box = get_eye_box(
                    face,
                    RIGHT_EYE,
                    width,
                    height,
                )

                if left_box:

                    x1, y1, x2, y2 = (
                        left_box
                    )

                    color = (
                        GREEN
                        if left_state == "OPEN"
                        else RED
                        if left_state == "CLOSED"
                        else YELLOW
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    # Anatomical label.
                    draw_text(
                        frame,
                        "YOUR LEFT",
                        (x1, max(25, y1 - 8)),
                        0.34,
                        color,
                        1,
                    )

                if right_box:

                    x1, y1, x2, y2 = (
                        right_box
                    )

                    color = (
                        GREEN
                        if right_state == "OPEN"
                        else RED
                        if right_state == "CLOSED"
                        else YELLOW
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    draw_text(
                        frame,
                        "YOUR RIGHT",
                        (x1, max(25, y1 - 8)),
                        0.34,
                        color,
                        1,
                    )

            else:

                left_state = "NO FACE"
                right_state = "NO FACE"

                drowsiness_state = "NO FACE"
                closed_duration = 0.0

                alarm.stop()

            # ====================================================
            # EYE ANALYSIS PANEL
            # ====================================================

            draw_text(
                frame,
                "EYE ANALYSIS",
                (
                    panel_x1 + 20,
                    panel_y1 + 165,
                ),
                0.55,
                WHITE,
                2,
            )

            draw_text(
                frame,
                "Anatomical eye labels",
                (
                    panel_x1 + 20,
                    panel_y1 + 190,
                ),
                0.34,
                MUTED,
            )

            draw_text(
                frame,
                "Your LEFT = screen RIGHT",
                (
                    panel_x1 + 20,
                    panel_y1 + 209,
                ),
                0.32,
                MUTED,
            )

            left_color = (
                GREEN
                if left_state == "OPEN"
                else RED
                if left_state == "CLOSED"
                else YELLOW
            )

            right_color = (
                GREEN
                if right_state == "OPEN"
                else RED
                if right_state == "CLOSED"
                else YELLOW
            )

            draw_text(
                frame,
                "YOUR LEFT EYE",
                (
                    panel_x1 + 20,
                    panel_y1 + 245,
                ),
                0.43,
                MUTED,
            )

            draw_text(
                frame,
                left_state,
                (
                    panel_x2 - 115,
                    panel_y1 + 245,
                ),
                0.46,
                left_color,
                2,
            )

            draw_text(
                frame,
                "YOUR RIGHT EYE",
                (
                    panel_x1 + 20,
                    panel_y1 + 278,
                ),
                0.43,
                MUTED,
            )

            draw_text(
                frame,
                right_state,
                (
                    panel_x2 - 115,
                    panel_y1 + 278,
                ),
                0.46,
                right_color,
                2,
            )

            draw_text(
                frame,
                f"L EAR {left_ear:.3f}",
                (
                    panel_x1 + 20,
                    panel_y1 + 312,
                ),
                0.37,
                WHITE,
            )

            draw_text(
                frame,
                f"R EAR {right_ear:.3f}",
                (
                    panel_x1 + 180,
                    panel_y1 + 312,
                ),
                0.37,
                WHITE,
            )

            draw_text(
                frame,
                f"L BLINK {left_blink:.2f}",
                (
                    panel_x1 + 20,
                    panel_y1 + 330,
                ),
                0.33,
                MUTED,
            )

            draw_text(
                frame,
                f"R BLINK {right_blink:.2f}",
                (
                    panel_x1 + 180,
                    panel_y1 + 330,
                ),
                0.33,
                MUTED,
            )

            both_closed_now = (
                calibrated
                and left_state == "CLOSED"
                and right_state == "CLOSED"
            )

            both_open_now = (
                calibrated
                and left_state == "OPEN"
                and right_state == "OPEN"
            )

            if both_closed_now:
                eye_hint = "BOTH EYES CLOSED"
                eye_hint_color = RED
            elif both_open_now:
                eye_hint = "BOTH EYES OPEN"
                eye_hint_color = GREEN
            else:
                eye_hint = "ONE EYE / TRANSITION"
                eye_hint_color = YELLOW

            draw_text(
                frame,
                eye_hint,
                (
                    panel_x1 + 20,
                    panel_y1 + 370,
                ),
                0.38,
                eye_hint_color,
                2,
            )

            if calibrated:

                draw_text(
                    frame,
                    (
                        f"L close/open "
                        f"{left_close:.3f}/"
                        f"{left_open:.3f}"
                    ),
                    (
                        panel_x1 + 20,
                        panel_y1 + 400,
                    ),
                    0.33,
                    MUTED,
                )

                draw_text(
                    frame,
                    (
                        f"R close/open "
                        f"{right_close:.3f}/"
                        f"{right_open:.3f}"
                    ),
                    (
                        panel_x1 + 180,
                        panel_y1 + 400,
                    ),
                    0.33,
                    MUTED,
                )

            else:

                elapsed = (
                    time.monotonic()
                    - calibration_start
                )

                remaining = max(
                    0.0,
                    CALIBRATION_SECONDS
                    - elapsed,
                )

                draw_text(
                    frame,
                    (
                        f"CALIBRATING "
                        f"{remaining:.1f}s"
                    ),
                    (
                        panel_x1 + 20,
                        panel_y1 + 400,
                    ),
                    0.38,
                    ORANGE,
                    1,
                )


            # ====================================================
            # DROWSINESS
            # ====================================================

            drowsy_y = panel_y1 + 455

            cv2.line(
                frame,
                (
                    panel_x1 + 20,
                    drowsy_y - 20,
                ),
                (
                    panel_x2 - 20,
                    drowsy_y - 20,
                ),
                BORDER,
                1,
            )

            draw_text(
                frame,
                "DROWSINESS STATUS",
                (
                    panel_x1 + 20,
                    drowsy_y + 15,
                ),
                0.52,
                WHITE,
                2,
            )

            status_color = {
                "NORMAL": GREEN,
                "BLINK": YELLOW,
                "DROWSY": ORANGE,
                "SLEEPING": RED,
            }.get(
                drowsiness_state,
                MUTED,
            )

            draw_text(
                frame,
                drowsiness_state,
                (
                    panel_x1 + 20,
                    drowsy_y + 50,
                ),
                0.55,
                status_color,
                2,
            )

            draw_text(
                frame,
                (
                    f"Both-eye closed "
                    f"{closed_duration:.2f}s"
                ),
                (
                    panel_x1 + 20,
                    drowsy_y + 78,
                ),
                0.36,
                WHITE,
            )

            draw_text(
                frame,
                f"Blink count {blink_count}",
                (
                    panel_x1 + 20,
                    drowsy_y + 105,
                ),
                0.36,
                WHITE,
            )

            alarm_text = (
                "ALARM: ON"
                if alarm.active
                else "Alarm: OFF"
            )

            draw_text(
                frame,
                alarm_text,
                (
                    panel_x1 + 20,
                    drowsy_y + 132,
                ),
                0.42,
                (
                    RED
                    if alarm.active
                    else MUTED
                ),
                2 if alarm.active else 1,
            )

            # ====================================================
            # SLEEPING ALERT
            # ====================================================

            if (
                drowsiness_state
                == "SLEEPING"
            ):

                overlay = frame.copy()

                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (width, height),
                    (0, 0, 120),
                    -1,
                )

                cv2.addWeighted(
                    overlay,
                    0.18,
                    frame,
                    0.82,
                    0,
                    frame,
                )

                cv2.rectangle(
                    frame,
                    (30, 105),
                    (
                        panel_x1 - 25,
                        190,
                    ),
                    RED,
                    2,
                )

                draw_text(
                    frame,
                    "SLEEPING / BOTH EYES CLOSED",
                    (55, 145),
                    0.65,
                    RED,
                    2,
                )

                draw_text(
                    frame,
                    (
                        f"Both eyes closed "
                        f"{closed_duration:.1f}s"
                    ),
                    (55, 175),
                    0.42,
                    WHITE,
                )

            # ====================================================
            # FOOTER
            # ====================================================

            draw_text(
                frame,
                f"FPS {fps:.1f}",
                (
                    28,
                    height - 25,
                ),
                0.40,
                MUTED,
            )

            draw_text(
                frame,
                "R Recalibrate",
                (
                    105,
                    height - 25,
                ),
                0.40,
                MUTED,
            )

            draw_text(
                frame,
                "Q Exit",
                (
                    225,
                    height - 25,
                ),
                0.40,
                MUTED,
            )

            cv2.imshow(
                "Sleeping Eye Detection | Proof of Concept",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # ----------------------------------------------------
            # RE-CALIBRATION
            # ----------------------------------------------------

            if key == ord("r"):

                calibrated = False

                calibration_start = (
                    time.monotonic()
                )

                left_calibration.clear()
                right_calibration.clear()

                left_history.clear()
                right_history.clear()

                left_state = "CALIBRATING"
                right_state = "CALIBRATING"

                left_candidate = "CALIBRATING"
                right_candidate = "CALIBRATING"

                left_candidate_frames = 0
                right_candidate_frames = 0

                engine.reset()
                alarm.stop()

                drowsiness_state = "NORMAL"
                closed_duration = 0.0

                print(
                    "\nRecalibration started. "
                    "Keep BOTH eyes OPEN.\n"
                )

    finally:

        alarm.close()

        camera.release()

        cv2.destroyAllWindows()

        face_landmarker.close()

        print("\nAlarm stopped.")
        print("Camera stopped.")
        print("Face Landmarker stopped.")
        print("Test completed.")


if __name__ == "__main__":
    main()