import cv2
import time
import os
import statistics
import threading
import winsound
from collections import deque
import mediapipe as mp
import numpy as np

try:
    from plyer import notification
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "face_landmarker.task"
)

CAMERA_INDEX = 0

FRAME_W, FRAME_H = 960, 540

WINDOW_NAME = "Sleeping Eye Detection | Proof of Concept"
WINDOW_W, WINDOW_H = 1400, 800

CALIBRATION_SECONDS = 3.0
SMOOTH_FRAMES = 3

# Per-eye visibility protection.  An eye whose landmarks become too
# compressed at a head angle is marked NOT VISIBLE instead of CLOSED.
# This is critical: the hidden/weak eye must NEVER inherit the state of
# the visible eye.
EYE_VISIBILITY_RATIO_MIN = 0.58
EYE_VISIBILITY_RATIO_RECOVER = 0.66

DROWSY_SECONDS = 0.8
SLEEPING_SECONDS = 2.0
ALARM_SECONDS = 0.8

NOTIFICATION_COOLDOWN = 10.0


# MediaPipe eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def _landmark_distance_2d(a, b):
    return (
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    ) ** 0.5


def _landmark_distance_3d(a, b):
    return (
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    ) ** 0.5


def eye_aspect_ratio(landmarks, indices):
    """Classic 2D EAR, retained for comparison/debugging."""
    p = [landmarks[i] for i in indices]
    h = _landmark_distance_2d(p[0], p[3])

    if h < 1e-6:
        return 0.0

    return (
        _landmark_distance_2d(p[1], p[5]) +
        _landmark_distance_2d(p[2], p[4])
    ) / (2.0 * h)


def eye_aspect_ratio_3d(landmarks, indices):
    """Pose-robust EAR using MediaPipe's 3D landmark coordinates."""
    p = [landmarks[i] for i in indices]
    h = _landmark_distance_3d(p[0], p[3])

    if h < 1e-6:
        return 0.0

    return (
        _landmark_distance_3d(p[1], p[5]) +
        _landmark_distance_3d(p[2], p[4])
    ) / (2.0 * h)


# ============================================================
# PER-EYE GEOMETRY / VISIBILITY
# ============================================================

def eye_geometry(landmarks, indices):
    """Return geometry used only to decide whether THIS eye is reliable."""
    p0 = landmarks[indices[0]]
    p3 = landmarks[indices[3]]

    # Eye corner-to-corner width.  This should remain reasonably stable
    # when an eye opens/closes, but becomes unreliable when the eye is
    # strongly foreshortened/occluded by a head turn.
    w2d = _landmark_distance_2d(p0, p3)
    w3d = _landmark_distance_3d(p0, p3)

    # Distance between the two outer eye corners.  It gives a scale that
    # changes much less than the width of the far eye during yaw.
    l = landmarks[263]
    r = landmarks[33]
    inter2d = _landmark_distance_2d(l, r)
    inter3d = _landmark_distance_3d(l, r)

    if inter2d < 1e-6 or inter3d < 1e-6:
        return 0.0, 0.0

    return w2d / inter2d, w3d / inter3d


# ============================================================
# EYE CLASSIFIER
# ============================================================

class EyeClassifier:
    """
    Per-eye open/closed classifier designed for the laptop-camera POC.

    Decision policy:
    - 3D EAR ratio is the main measurement.
    - 2D EAR ratio is a confirmation signal for a real closure.
    - MediaPipe blink/eye blendshape is NEVER allowed to close an eye.
    - A short temporal requirement prevents one bad landmark frame from
      changing OPEN -> CLOSED.
    - Hysteresis makes CLOSED -> OPEN require clear evidence of openness.

    This is intentionally per-eye, so closing only the left eye does not
    close the right eye and vice versa.
    """

    # Normalized ratios relative to the user's calibrated open-eye baseline.
    # An open eye in the current laptop-camera tests is commonly around
    # 0.75 or higher, while a genuinely closed eye should fall much lower.
    CLOSE_RATIO_3D = 0.54
    OPEN_RATIO_3D = 0.62

    # 2D EAR is used only as confirmation. It is more sensitive to head pose,
    # so it is deliberately not used alone except for a very strong collapse.
    CLOSE_RATIO_2D = 0.70
    STRONG_CLOSE_RATIO_2D = 0.58

    CLOSE_CONFIRM_FRAMES = 2
    OPEN_CONFIRM_FRAMES = 2

    def __init__(self):
        self.base_ear = 0.25
        self.base_ear_2d = 0.25
        self.base_blink = 0.0
        self.base_eye_width_2d = 0.30
        self.base_eye_width_3d = 0.30

        self.ear_history = deque(maxlen=SMOOTH_FRAMES)
        self.ear_2d_history = deque(maxlen=SMOOTH_FRAMES)
        self.blink_history = deque(maxlen=SMOOTH_FRAMES)
        self.closed_history = deque(maxlen=SMOOTH_FRAMES)

        self.closed_state = False
        self.close_votes = 0
        self.open_votes = 0

    @staticmethod
    def _robust_open_baseline(values, minimum=0.08):
        """
        Estimate an open-eye baseline while reducing the effect of blinks.

        Calibration is performed while the user keeps both eyes open, but
        short blinks can still occur. We discard the lowest 20% of samples
        before taking the median.
        """
        if not values:
            return minimum

        clean = [float(v) for v in values if v > 0.0]
        if not clean:
            return minimum

        clean.sort()
        cut = int(len(clean) * 0.20)

        if len(clean) >= 10 and cut > 0:
            clean = clean[cut:]

        return max(minimum, statistics.median(clean))

    def calibrate(self, ears, blinks, ears_2d=None, eye_widths_2d=None, eye_widths_3d=None):
        if ears:
            self.base_ear = self._robust_open_baseline(ears)

        if ears_2d:
            self.base_ear_2d = self._robust_open_baseline(ears_2d)

        if blinks:
            self.base_blink = max(0.0, statistics.median(blinks))

        if eye_widths_2d:
            self.base_eye_width_2d = self._robust_open_baseline(eye_widths_2d, minimum=0.05)

        if eye_widths_3d:
            self.base_eye_width_3d = self._robust_open_baseline(eye_widths_3d, minimum=0.05)

        # Always start the calibrated eye as OPEN.
        self.closed_state = False
        self.close_votes = 0
        self.open_votes = self.OPEN_CONFIRM_FRAMES
        self.closed_history.clear()

    def update(self, ear_3d, ear_2d, blink, head_pitch=0.0, eye_width_2d=None, eye_width_3d=None):

        self.ear_history.append(float(ear_3d))
        self.ear_2d_history.append(float(ear_2d))
        self.blink_history.append(float(blink))

        # ------------------------------------------------------------
        # THIS EYE'S OWN VISIBILITY
        # ------------------------------------------------------------
        # Never infer one eye from the other.  When the head turns,
        # MediaPipe can still return landmarks for the far eye even though
        # those landmarks are no longer reliable.  Such landmarks can
        # produce a tiny EAR and falsely report CLOSED.
        width2d = float(eye_width_2d) if eye_width_2d is not None else self.base_eye_width_2d
        width3d = float(eye_width_3d) if eye_width_3d is not None else self.base_eye_width_3d

        width_ratio_2d = width2d / max(self.base_eye_width_2d, 1e-6)
        width_ratio_3d = width3d / max(self.base_eye_width_3d, 1e-6)

        eye_visible = (
            width_ratio_2d >= EYE_VISIBILITY_RATIO_MIN
            and width_ratio_3d >= EYE_VISIBILITY_RATIO_MIN
        )

        # If the geometry is borderline, require stronger evidence from
        # both dimensions before accepting the eye as visible.
        eye_visible_strong = (
            width_ratio_2d >= EYE_VISIBILITY_RATIO_RECOVER
            or width_ratio_3d >= EYE_VISIBILITY_RATIO_RECOVER
        )

        if not eye_visible and not eye_visible_strong:
            self.closed_state = False
            self.close_votes = 0
            self.open_votes = self.OPEN_CONFIRM_FRAMES
            self.closed_history.clear()

            return {
                "closed": False,
                "visible": False,
                "ear": statistics.median(self.ear_history),
                "ear_2d": statistics.median(self.ear_2d_history),
                "ear_ratio": statistics.median(self.ear_history) / max(self.base_ear, 1e-6),
                "ear_2d_ratio": statistics.median(self.ear_2d_history) / max(self.base_ear_2d, 1e-6),
                "blink": statistics.median(self.blink_history),
                "blink_peak": max(self.blink_history) if self.blink_history else 0.0,
                "ear_threshold": max(0.055, self.base_ear * self.CLOSE_RATIO_3D),
                "open_threshold": max(0.055, self.base_ear * self.OPEN_RATIO_3D),
                "blink_threshold": max(0.65, self.base_blink + 0.42),
                "head_pitch": head_pitch,
                "blink_supported": False,
                "blink_event": False,
                "raw_closed": False,
                "raw_open": False,
                "close_votes": 0,
                "open_votes": self.OPEN_CONFIRM_FRAMES,
                "width_ratio_2d": width_ratio_2d,
                "width_ratio_3d": width_ratio_3d
            }

        smooth_ear = statistics.median(self.ear_history)
        smooth_ear_2d = statistics.median(self.ear_2d_history)
        smooth_blink = statistics.median(self.blink_history)
        # Keep the strongest recent blink signal separately. A median over
        # several frames can hide a short first blink because most frames
        # around the blink are still OPEN.
        recent_blink_peak = max(self.blink_history) if self.blink_history else 0.0

        # Normalize each eye against its own calibrated open-eye baseline.
        ear_ratio = smooth_ear / max(self.base_ear, 1e-6)
        ear_2d_ratio = smooth_ear_2d / max(self.base_ear_2d, 1e-6)

        # Absolute thresholds are useful diagnostics and protect against a
        # pathological calibration baseline.
        close_threshold = max(0.055, self.base_ear * self.CLOSE_RATIO_3D)
        open_threshold = max(
            close_threshold + 0.012,
            self.base_ear * self.OPEN_RATIO_3D
        )

        # Blink score is diagnostic ONLY. It never controls eye state.
        blink_threshold = max(0.65, self.base_blink + 0.42)

        # ------------------------------------------------------------
        # CLOSED EVIDENCE
        # ------------------------------------------------------------
        # Normal closure should reduce both geometric measurements.
        # This prevents head-angle changes that affect only one EAR from
        # falsely closing an eye.
        # A real closed eye should show a substantial reduction in BOTH
        # measurements. Requiring agreement reduces false closures caused by
        # head tilt, perspective, or one unstable landmark set.
        geometric_close = (
            ear_ratio <= self.CLOSE_RATIO_3D
            and ear_2d_ratio <= self.CLOSE_RATIO_2D
        )

        # Very strong 2D collapse can confirm closure when 3D depth is noisy,
        # but it still requires the 3D ratio to be clearly below normal-open.
        strong_2d_close = (
            ear_2d_ratio <= self.STRONG_CLOSE_RATIO_2D
            and ear_ratio <= 0.58
        )

        raw_closed = geometric_close or strong_2d_close

        # ------------------------------------------------------------
        # TRANSIENT BLINK EVENT
        # ------------------------------------------------------------
        # A normal blink is much shorter than drowsiness.  We use the
        # MediaPipe blink blendshape only as a short visual event, combined
        # with a geometric eye-opening drop.  It NEVER changes the persistent
        # closed_state and therefore cannot start the drowsiness alarm.
        blink_event_threshold = max(
            0.55,
            self.base_blink + 0.30
        )

        # Use the PEAK of the recent blink signal so the first short blink
        # is not swallowed by the median smoothing window. Geometry is still
        # required, so a blink score alone cannot turn the eye red.
        blink_event = (
            recent_blink_peak >= blink_event_threshold
            and (
                ear_ratio <= 0.90
                or ear_2d_ratio <= 0.95
            )
        )

        # ------------------------------------------------------------
        # OPEN EVIDENCE
        # ------------------------------------------------------------
        # Either measurement can provide recovery evidence. This is useful
        # when head movement temporarily changes one EAR value.
        raw_open = (
            ear_ratio >= self.OPEN_RATIO_3D
            or ear_2d_ratio >= 0.72
        )

        # Recovery band for open eyes during head pose changes. It is only
        # used when there is no simultaneous strong closure evidence.
        recovery_open = (
            ear_ratio >= 0.60
            and ear_2d_ratio >= 0.64
            and not raw_closed
        )
        raw_open = raw_open or recovery_open

        # ------------------------------------------------------------
        # TEMPORAL HYSTERESIS
        # ------------------------------------------------------------
        if raw_closed:
            self.close_votes += 1
        else:
            self.close_votes = max(0, self.close_votes - 1)

        if raw_open:
            self.open_votes += 1
        else:
            self.open_votes = max(0, self.open_votes - 1)

        if not self.closed_state:
            # Need several consecutive closure frames before reporting
            # CLOSED. A single bad frame therefore cannot trigger it.
            if self.close_votes >= self.CLOSE_CONFIRM_FRAMES:
                self.closed_state = True
                self.open_votes = 0
        else:
            # Need clear open evidence before returning to OPEN.
            if self.open_votes >= self.OPEN_CONFIRM_FRAMES:
                self.closed_state = False
                self.close_votes = 0

        self.closed_history.append(self.closed_state)

        # The classifier state is already temporally filtered. Keep the
        # short history for diagnostics and additional stability.
        votes = sum(1 for x in self.closed_history if x)
        closed = (
            votes >= max(2, len(self.closed_history) - 1)
            if self.closed_history
            else self.closed_state
        )

        return {
            "closed": closed,
            "visible": True,
            "ear": smooth_ear,
            "ear_2d": smooth_ear_2d,
            "ear_ratio": ear_ratio,
            "ear_2d_ratio": ear_2d_ratio,
            "blink": smooth_blink,
            "blink_peak": recent_blink_peak,
            "ear_threshold": close_threshold,
            "open_threshold": open_threshold,
            "blink_threshold": blink_threshold,
            "head_pitch": head_pitch,
            "blink_supported": (
                smooth_blink >= blink_threshold
                and raw_closed
            ),
            "blink_event": blink_event,
            "raw_closed": raw_closed,
            "raw_open": raw_open,
            "close_votes": self.close_votes,
            "open_votes": self.open_votes,
            "width_ratio_2d": width_ratio_2d,
            "width_ratio_3d": width_ratio_3d
        }


# ============================================================
# MEDIAPIPE BLENDSHAPES
# ============================================================

def get_blendshapes(result):

    left = 0.0
    right = 0.0

    if not result.face_blendshapes:
        return left, right

    for category in result.face_blendshapes[0]:

        name = category.category_name or ""

        if name == "eyeBlinkLeft":

            left = float(
                category.score
            )

        elif name == "eyeBlinkRight":

            right = float(
                category.score
            )

    return left, right


# ============================================================
# TEXT DRAWING
# ============================================================

def draw_text(
    img,
    value,
    xy,
    scale=0.5,
    color=(255, 255, 255),
    thickness=1
):

    cv2.putText(
        img,
        value,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# EYE BOX
# ============================================================

def eye_box(
    landmarks,
    indices,
    width,
    height
):

    xs = [
        landmarks[i].x * width
        for i in indices
    ]

    ys = [
        landmarks[i].y * height
        for i in indices
    ]

    x1 = int(min(xs))
    x2 = int(max(xs))

    y1 = int(min(ys))
    y2 = int(max(ys))

    px = 8
    py = 8

    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(width - 1, x2 + px),
        min(height - 1, y2 + py)
    )


# ============================================================
# HEAD POSE ESTIMATION
# ============================================================

def estimate_head_pitch(landmarks, width, height):
    """
    Estimate approximate head pitch in degrees.

    The value is used only as a protection signal for the eye
    classifier. Exact 3D head-pose accuracy is not required.
    """

    # Stable facial landmarks:
    # nose tip, chin, outer eye corners and mouth corners.
    landmark_ids = [
        1,    # nose tip
        152,  # chin
        263,  # left eye outer corner
        33,   # right eye outer corner
        287,  # left mouth corner
        57    # right mouth corner
    ]

    try:
        image_points = []

        for idx in landmark_ids:
            lm = landmarks[idx]
            image_points.append([
                float(lm.x * width),
                float(lm.y * height)
            ])

        image_points = np.array(
            image_points,
            dtype=np.float64
        )

        # Approximate generic 3D face model in millimetres.
        model_points = np.array([
            [0.0,   0.0,    0.0],     # nose
            [0.0,  -63.0, -12.0],     # chin
            [-43.0, 32.0, -26.0],     # left eye outer
            [43.0,  32.0, -26.0],     # right eye outer
            [-32.0, -28.0, -20.0],    # left mouth
            [32.0,  -28.0, -20.0],     # right mouth
        ], dtype=np.float64)

        focal_length = float(width)

        camera_matrix = np.array([
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64
        )

        success, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        _, _, _, _, _, _, euler_angles = cv2.RQDecomp3x3(
            rotation_matrix
        )

        pitch = float(euler_angles[0])

        if not (-90.0 <= pitch <= 90.0):
            return 0.0

        return pitch

    except Exception:
        return 0.0


# ============================================================
# ALARM MANAGER
# ============================================================

class AlarmManager:

    def __init__(self):

        self.alarm_active = False

        self.stop_event = threading.Event()

        self.thread = None

        self.last_notification = 0.0

        self.notification_sent_for_closure = False

    # --------------------------------------------------------
    # START ALARM
    # --------------------------------------------------------

    def start(self):

        if not self.alarm_active:

            self.alarm_active = True

            self.stop_event.clear()

            self.thread = threading.Thread(
                target=self._alarm_loop,
                daemon=True
            )

            self.thread.start()

        if not self.notification_sent_for_closure:

            self.send_notification()

            self.notification_sent_for_closure = True

    # --------------------------------------------------------
    # CUSTOM EMERGENCY SIREN
    # --------------------------------------------------------

    def _alarm_loop(self):

        # ====================================================
        # EMERGENCY SIREN - OPTION A
        #
        # BEEP - BEEP - BEEP - LONG BEEP
        # then repeat
        # ====================================================

        while not self.stop_event.is_set():

            try:

                # -------------------------------
                # First beep
                # -------------------------------

                winsound.Beep(
                    900,
                    180
                )

                if self.stop_event.is_set():
                    break

                time.sleep(0.08)

                # -------------------------------
                # Second beep
                # -------------------------------

                winsound.Beep(
                    1200,
                    180
                )

                if self.stop_event.is_set():
                    break

                time.sleep(0.08)

                # -------------------------------
                # Third beep
                # -------------------------------

                winsound.Beep(
                    900,
                    180
                )

                if self.stop_event.is_set():
                    break

                time.sleep(0.20)

                # -------------------------------
                # Higher warning tone
                # -------------------------------

                winsound.Beep(
                    1400,
                    220
                )

                if self.stop_event.is_set():
                    break

                # -------------------------------
                # Pause before repeating
                # -------------------------------

                self.stop_event.wait(
                    0.45
                )

            except Exception as e:

                print(
                    f"Alarm sound error: {e}"
                )

                try:

                    winsound.MessageBeep(
                        winsound.MB_ICONHAND
                    )

                except Exception:
                    pass

                self.stop_event.wait(
                    0.5
                )

    # --------------------------------------------------------
    # STOP ALARM
    # --------------------------------------------------------

    def stop(self):

        self.alarm_active = False

        self.stop_event.set()

    # --------------------------------------------------------
    # NEW CLOSURE
    # --------------------------------------------------------

    def new_closure(self):

        self.stop()

        self.notification_sent_for_closure = False

    # --------------------------------------------------------
    # WINDOWS NOTIFICATION
    # --------------------------------------------------------

    def send_notification(self):

        now = time.monotonic()

        if (
            now - self.last_notification
            < NOTIFICATION_COOLDOWN
        ):
            return

        self.last_notification = now

        if NOTIFICATION_AVAILABLE:

            try:

                notification.notify(
                    title="DROWSINESS ALERT",
                    message=(
                        "Both eyes have been closed "
                        "for too long. Please stay alert."
                    ),
                    app_name="Sleeping Eye Detection",
                    timeout=5
                )

                print(
                    "Windows notification sent."
                )

            except Exception as e:

                print(
                    f"Notification error: {e}"
                )

        else:

            print(
                "DROWSINESS ALERT - "
                "install plyer: pip install plyer"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Face Landmarker model not found:\n"
            f"{MODEL_PATH}"
        )

    # ========================================================
    # CAMERA
    # ========================================================

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
            "Camera could not be opened. "
            "Check Windows Camera permissions."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_W
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_H
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

    # ========================================================
    # MEDIAPIPE
    # ========================================================

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

        output_face_blendshapes=True
    )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    left_classifier = EyeClassifier()

    right_classifier = EyeClassifier()

    calibration = True

    calibration_start = (
        time.monotonic()
    )

    left_ears = []
    right_ears = []
    left_ears_2d = []
    right_ears_2d = []

    left_blinks = []
    right_blinks = []

    left_eye_widths_2d = []
    right_eye_widths_2d = []
    left_eye_widths_3d = []
    right_eye_widths_3d = []

    closed_start = None

    previous_both_closed = False

    blink_count = 0

    # Independent blink-event tracking. Each eye is tracked separately;
    # one eye can never create a blink event for the other eye.
    left_eye_closed_start = None
    right_eye_closed_start = None
    previous_left_eye_closed = False
    previous_right_eye_closed = False
    previous_left_blink_event = False
    previous_right_blink_event = False
    last_blink_count_time = -999.0
    BLINK_MIN_SECONDS = 0.04
    BLINK_MAX_SECONDS = 0.75
    BLINK_MERGE_SECONDS = 0.25

    state = "CALIBRATING"

    alarm_manager = AlarmManager()

    program_start = (
        time.monotonic()
    )

    frame_count = 0

    last_timestamp = -1

    # ========================================================
    # CONSOLE
    # ========================================================

    print("=" * 60)

    print(
        "SLEEPING EYE DETECTION V6 - INDEPENDENT EYE ANGLE SAFE"
    )

    print(
        "Keep BOTH eyes OPEN for 3 seconds."
    )

    print(
        "Both eyes closed > 0.8s = "
        "DROWSY + ALARM + NOTIFICATION"
    )

    print(
        "Both independently visible eyes closed > 2.0s = SLEEPING"
    )

    print(
        "A poorly visible eye at a head angle is NOT VISIBLE, never CLOSED."
    )

    print(
        "R = recalibrate | Q = quit"
    )

    print("=" * 60)

    # ========================================================
    # WINDOW
    # ========================================================

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_W,
        WINDOW_H
    )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        with FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                # =================================================
                # CAMERA FRAME
                # =================================================

                success, frame = (
                    camera.read()
                )

                if not success:
                    continue

                frame = cv2.flip(
                    frame,
                    1
                )

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

                now = time.monotonic()

                timestamp_ms = int(
                    (
                        now -
                        program_start
                    ) * 1000
                )

                if (
                    timestamp_ms
                    <= last_timestamp
                ):

                    timestamp_ms = (
                        last_timestamp + 1
                    )

                last_timestamp = (
                    timestamp_ms
                )

                # =================================================
                # FACE DETECTION
                # =================================================

                result = (
                    landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

                frame_count += 1

                display = frame.copy()

                height, width = (
                    display.shape[:2]
                )

                left_data = None
                right_data = None

                # =================================================
                # FACE FOUND
                # =================================================

                if result.face_landmarks:

                    landmarks = (
                        result.face_landmarks[0]
                    )

                    # ------------------------------------------------
                    # EAR
                    # ------------------------------------------------

                    left_ear_2d = eye_aspect_ratio(
                        landmarks,
                        LEFT_EYE
                    )

                    right_ear_2d = eye_aspect_ratio(
                        landmarks,
                        RIGHT_EYE
                    )

                    left_ear = eye_aspect_ratio_3d(
                        landmarks,
                        LEFT_EYE
                    )

                    right_ear = eye_aspect_ratio_3d(
                        landmarks,
                        RIGHT_EYE
                    )

                    # ------------------------------------------------
                    # PER-EYE VISIBILITY GEOMETRY
                    # ------------------------------------------------
                    left_width_2d, left_width_3d = eye_geometry(
                        landmarks, LEFT_EYE
                    )
                    right_width_2d, right_width_3d = eye_geometry(
                        landmarks, RIGHT_EYE
                    )

                    # ------------------------------------------------
                    # BLINK
                    # ------------------------------------------------

                    (
                        left_blink,
                        right_blink
                    ) = get_blendshapes(
                        result
                    )

                    # ------------------------------------------------
                    # HEAD POSE
                    # ------------------------------------------------
                    # Used only to protect eye classification when
                    # the head is strongly raised/lowered.

                    head_pitch = estimate_head_pitch(
                        landmarks,
                        width,
                        height
                    )

                    # =================================================
                    # CALIBRATION
                    # =================================================

                    if calibration:

                        left_ears.append(left_ear)
                        right_ears.append(right_ear)
                        left_ears_2d.append(left_ear_2d)
                        right_ears_2d.append(right_ear_2d)

                        left_blinks.append(
                            left_blink
                        )

                        right_blinks.append(
                            right_blink
                        )

                        left_eye_widths_2d.append(left_width_2d)
                        right_eye_widths_2d.append(right_width_2d)
                        left_eye_widths_3d.append(left_width_3d)
                        right_eye_widths_3d.append(right_width_3d)

                        if (
                            now -
                            calibration_start
                            >= CALIBRATION_SECONDS
                            and
                            len(left_ears)
                            >= 20
                        ):

                            left_classifier.calibrate(
                                left_ears,
                                left_blinks,
                                left_ears_2d
                            )

                            right_classifier.calibrate(
                                right_ears,
                                right_blinks,
                                right_ears_2d
                            )

                            calibration = False

                            state = "NORMAL"

                            print(
                                "\nCALIBRATION COMPLETE"
                            )

                            print(
                                "Left EAR baseline : "
                                f"{left_classifier.base_ear:.3f}"
                            )

                            print(
                                "Right EAR baseline: "
                                f"{right_classifier.base_ear:.3f}"
                            )

                            print(
                                "Left blink base   : "
                                f"{left_classifier.base_blink:.3f}"
                            )

                            print(
                                "Right blink base  : "
                                f"{right_classifier.base_blink:.3f}\n"
                            )

                    # =================================================
                    # EYE ANALYSIS
                    # =================================================

                    else:

                        left_data = (
                            left_classifier.update(
                                left_ear,
                                left_ear_2d,
                                left_blink,
                                head_pitch,
                                left_width_2d,
                                left_width_3d
                            )
                        )

                        right_data = (
                            right_classifier.update(
                                right_ear,
                                right_ear_2d,
                                right_blink,
                                head_pitch,
                                right_width_2d,
                                right_width_3d
                            )
                        )

                        # ------------------------------------------------
                        # FAST, INDEPENDENT PER-EYE BLINK COUNTING
                        # ------------------------------------------------
                        # Do NOT wait for the persistent CLOSED state to
                        # transition back to OPEN. A very fast blink may last
                        # only a few camera frames, while the eye classifier
                        # intentionally requires several frames before it
                        # changes its persistent CLOSED state.
                        #
                        # Therefore the blink counter uses the short
                        # blink_event generated by each eye independently.
                        # A rising edge = one blink. A cooldown merges a
                        # simultaneous left+right blink into one event.
                        #
                        # Crucially, LEFT and RIGHT are completely separate:
                        # one eye can never create a blink for the other eye.

                        left_visible = left_data.get("visible", False)
                        right_visible = right_data.get("visible", False)

                        left_closed_now = (
                            left_visible and left_data["closed"]
                        )
                        right_closed_now = (
                            right_visible and right_data["closed"]
                        )

                        left_blink_event = (
                            left_visible
                            and left_data.get("blink_event", False)
                        )
                        right_blink_event = (
                            right_visible
                            and right_data.get("blink_event", False)
                        )

                        # Count the LEFT blink immediately when the blink
                        # event starts. This works even for a very fast blink.
                        if (
                            left_blink_event
                            and not previous_left_blink_event
                        ):
                            if (
                                now - last_blink_count_time
                                >= BLINK_MERGE_SECONDS
                            ):
                                blink_count += 1
                                last_blink_count_time = now

                        # Count the RIGHT blink independently.
                        if (
                            right_blink_event
                            and not previous_right_blink_event
                        ):
                            if (
                                now - last_blink_count_time
                                >= BLINK_MERGE_SECONDS
                            ):
                                blink_count += 1
                                last_blink_count_time = now

                        # Keep the closure timers for drowsiness/alarm logic.
                        # These timers are NOT used for blink counting.
                        if (
                            left_closed_now
                            and not previous_left_eye_closed
                        ):
                            left_eye_closed_start = now

                        if (
                            right_closed_now
                            and not previous_right_eye_closed
                        ):
                            right_eye_closed_start = now

                        # Clear closure timers when the corresponding eye
                        # reopens. Long closures remain drowsiness events,
                        # not blink events.
                        if (
                            previous_left_eye_closed
                            and not left_closed_now
                        ):
                            left_eye_closed_start = None

                        if (
                            previous_right_eye_closed
                            and not right_closed_now
                        ):
                            right_eye_closed_start = None

                        # ------------------------------------------------
                        # BOTH EYES CLOSED -- STILL ONLY WHEN BOTH ARE
                        # INDEPENDENTLY VISIBLE AND INDEPENDENTLY CLOSED
                        # ------------------------------------------------
                        both_closed = (
                            left_visible
                            and right_visible
                            and left_data["closed"]
                            and right_data["closed"]
                        )

                        if both_closed and not previous_both_closed:
                            closed_start = now
                            alarm_manager.new_closure()

                        closed_duration = (
                            now - closed_start
                            if closed_start is not None
                            else 0.0
                        )

                        if (
                            not both_closed
                            and previous_both_closed
                            and closed_start is not None
                        ):
                            closed_start = None
                            alarm_manager.new_closure()

                        # =================================================
                        # DROWSINESS STATE
                        # =================================================

                        if both_closed:

                            if (
                                closed_duration
                                <
                                DROWSY_SECONDS
                            ):

                                state = "BLINK"

                            elif (
                                closed_duration
                                <
                                SLEEPING_SECONDS
                            ):

                                state = "DROWSY"

                            else:

                                state = "SLEEPING"

                            # ------------------------------------------------
                            # START ALARM
                            # ------------------------------------------------

                            if (
                                closed_duration
                                >=
                                ALARM_SECONDS
                            ):

                                alarm_manager.start()

                        else:

                            # A short blink is shown as BLINK, but because
                            # blink_event does not change `both_closed`, it
                            # cannot start the drowsiness/alarm timer.
                            if (
                                left_data.get("blink_event", False)
                                or
                                right_data.get("blink_event", False)
                            ):
                                state = "BLINK"
                            else:
                                state = "NORMAL"

                            alarm_manager.new_closure()

                        previous_left_eye_closed = left_closed_now
                        previous_right_eye_closed = right_closed_now
                        previous_left_blink_event = left_blink_event
                        previous_right_blink_event = right_blink_event

                        previous_both_closed = (
                            both_closed
                        )

                    # =================================================
                    # EYE BOXES
                    # =================================================

                    left_box = eye_box(
                        landmarks,
                        LEFT_EYE,
                        width,
                        height
                    )

                    right_box = eye_box(
                        landmarks,
                        RIGHT_EYE,
                        width,
                        height
                    )

                    # ------------------------------------------------
                    # EYE COLORS
                    # ------------------------------------------------

                    if left_data is None:

                        left_color = (
                            0,
                            255,
                            255
                        )

                        right_color = (
                            0,
                            255,
                            255
                        )

                    else:

                        if not left_data.get("visible", True):
                            left_color = (0, 215, 255)
                        else:
                            left_visual_closed = (
                                left_data["closed"]
                                or
                                left_data.get("blink_event", False)
                            )
                            left_color = (0, 0, 255) if left_visual_closed else (0, 255, 0)

                        if not right_data.get("visible", True):
                            right_color = (0, 215, 255)
                        else:
                            right_visual_closed = (
                                right_data["closed"]
                                or
                                right_data.get("blink_event", False)
                            )
                            right_color = (0, 0, 255) if right_visual_closed else (0, 255, 0)

                    # ------------------------------------------------
                    # DRAW EYE BOXES
                    # ------------------------------------------------

                    cv2.rectangle(
                        display,
                        left_box[:2],
                        left_box[2:],
                        left_color,
                        2
                    )

                    cv2.rectangle(
                        display,
                        right_box[:2],
                        right_box[2:],
                        right_color,
                        2
                    )

                    # ------------------------------------------------
                    # EYE LABELS
                    # ------------------------------------------------

                    draw_text(
                        display,
                        "LEFT EYE",
                        (
                            left_box[0],
                            max(
                                20,
                                left_box[1] - 8
                            )
                        ),
                        0.38,
                        left_color,
                        1
                    )

                    draw_text(
                        display,
                        "RIGHT EYE",
                        (
                            right_box[0],
                            max(
                                20,
                                right_box[1] - 8
                            )
                        ),
                        0.38,
                        right_color,
                        1
                    )

                # =================================================
                # NO FACE
                # =================================================

                else:

                    state = "NO FACE"

                    closed_start = None

                    previous_both_closed = False

                    alarm_manager.new_closure()

                # =================================================
                # HUD PANEL
                # =================================================

                # Narrower panel
                panel_width = 230

                panel_x = max(
                    10,
                    width - panel_width
                )

                hud = display.copy()

                cv2.rectangle(
                    hud,
                    (panel_x, 10),
                    (width - 10, height - 10),
                    (16, 17, 19),
                    -1
                )

                display = cv2.addWeighted(
                    hud,
                    0.78,
                    display,
                    0.22,
                    0
                )

                # =================================================
                # TITLE
                # =================================================

                draw_text(
                    display,
                    "SLEEPING EYE DETECTION",
                    (18, 30),
                    0.56,
                    (245, 245, 245),
                    1
                )

                draw_text(
                    display,
                    "3D EAR + BLINK  |  ROBUST MODE",
                    (18, 52),
                    0.32,
                    (150, 150, 150),
                    1
                )

                # =================================================
                # SYSTEM STATUS
                # =================================================

                y = 40

                draw_text(
                    display,
                    "SYSTEM STATUS",
                    (
                        panel_x + 15,
                        y
                    ),
                    0.46,
                    (245, 245, 245),
                    1
                )

                draw_text(
                    display,
                    "● CAMERA       READY",
                    (
                        panel_x + 15,
                        y + 28
                    ),
                    0.34,
                    (70, 230, 90),
                    1
                )

                face_status = (
                    "ACTIVE"
                    if result.face_landmarks
                    else
                    "WAITING"
                )

                face_color = (
                    (70, 230, 90)
                    if result.face_landmarks
                    else
                    (0, 165, 255)
                )

                draw_text(
                    display,
                    f"● FACE TRACKING  {face_status}",
                    (
                        panel_x + 15,
                        y + 52
                    ),
                    0.34,
                    face_color,
                    1
                )

                # =================================================
                # EYE ANALYSIS
                # =================================================

                y = 125

                draw_text(
                    display,
                    "EYE ANALYSIS",
                    (
                        panel_x + 15,
                        y
                    ),
                    0.46,
                    (245, 245, 245),
                    1
                )

                # =================================================
                # CALIBRATION DISPLAY
                # =================================================

                if calibration:

                    remaining = max(
                        0.0,
                        CALIBRATION_SECONDS -
                        (
                            now -
                            calibration_start
                        )
                    )

                    draw_text(
                        display,
                        "CALIBRATING...",
                        (
                            panel_x + 15,
                            y + 32
                        ),
                        0.5,
                        (0, 210, 255),
                        2
                    )

                    draw_text(
                        display,
                        "KEEP BOTH EYES OPEN",
                        (
                            panel_x + 15,
                            y + 58
                        ),
                        0.38,
                        (220, 220, 220),
                        1
                    )

                    draw_text(
                        display,
                        f"Ready in {remaining:.1f}s",
                        (
                            panel_x + 15,
                            y + 80
                        ),
                        0.38,
                        (220, 220, 220),
                        1
                    )

                # =================================================
                # EYE DATA
                # =================================================

                elif left_data and right_data:

                    ls = (
                        "CLOSED" if left_data["closed"]
                        else ("OPEN" if left_data.get("visible", True) else "NOT VISIBLE")
                    )

                    rs = (
                        "CLOSED" if right_data["closed"]
                        else ("OPEN" if right_data.get("visible", True) else "NOT VISIBLE")
                    )

                    lc = (
                        (0, 0, 255) if left_data["closed"]
                        else ((0, 215, 255) if not left_data.get("visible", True) else (70, 230, 90))
                    )

                    rc = (
                        (0, 0, 255) if right_data["closed"]
                        else ((0, 215, 255) if not right_data.get("visible", True) else (70, 230, 90))
                    )

                    # ------------------------------------------------
                    # LEFT EYE
                    # ------------------------------------------------

                    draw_text(
                        display,
                        f"LEFT EYE  : {ls}",
                        (
                            panel_x + 15,
                            y + 32
                        ),
                        0.36,
                        lc,
                        1
                    )

                    # ------------------------------------------------
                    # RIGHT EYE
                    # ------------------------------------------------

                    draw_text(
                        display,
                        f"RIGHT EYE : {rs}",
                        (
                            panel_x + 15,
                            y + 57
                        ),
                        0.36,
                        rc,
                        1
                    )

                    # ------------------------------------------------
                    # EAR / BLINK VALUES
                    # ------------------------------------------------

                    draw_text(
                        display,
                        (
                            f"L EAR {left_data['ear']:.3f} "
                            f"L BLINK {left_data['blink']:.2f}"
                        ),
                        (
                            panel_x + 15,
                            y + 84
                        ),
                        0.34,
                        (215, 215, 215),
                        1
                    )

                    draw_text(
                        display,
                        (
                            f"R EAR {right_data['ear']:.3f} "
                            f"R BLINK {right_data['blink']:.2f}"
                        ),
                        (
                            panel_x + 15,
                            y + 106
                        ),
                        0.34,
                        (215, 215, 215),
                        1
                    )

                    draw_text(
                        display,
                        (
                            f"EAR close thresholds "
                            f"{left_data['ear_threshold']:.3f} / "
                            f"{right_data['ear_threshold']:.3f}"
                        ),
                        (
                            panel_x + 15,
                            y + 128
                        ),
                        0.31,
                        (155, 155, 155),
                        1
                    )

                    draw_text(
                        display,
                        (
                            f"EAR ratio "
                            f"{left_data['ear_ratio']:.2f} / "
                            f"{right_data['ear_ratio']:.2f}"
                        ),
                        (
                            panel_x + 15,
                            y + 148
                        ),
                        0.30,
                        (155, 155, 155),
                        1
                    )

                    draw_text(
                        display,
                        (
                            f"Eye visibility "
                            f"{left_data['width_ratio_2d']:.2f} / "
                            f"{right_data['width_ratio_2d']:.2f}"
                        ),
                        (
                            panel_x + 15,
                            y + 168
                        ),
                        0.29,
                        (155, 155, 155),
                        1
                    )

                    # =================================================
                    # DROWSINESS STATUS
                    # =================================================

                    sy = y + 202

                    draw_text(
                        display,
                        "DROWSINESS STATUS",
                        (
                            panel_x + 15,
                            sy
                        ),
                        0.46,
                        (245, 245, 245),
                        1
                    )

                    colors = {

                        "NORMAL":
                            (70, 230, 90),

                        "BLINK":
                            (0, 220, 255),

                        "DROWSY":
                            (0, 165, 255),

                        "SLEEPING":
                            (0, 0, 255),

                        "NO FACE":
                            (0, 165, 255)
                    }

                    draw_text(
                        display,
                        state,
                        (
                            panel_x + 15,
                            sy + 33
                        ),
                        0.52,
                        colors.get(
                            state,
                            (255, 255, 255)
                        ),
                        1
                    )

                    cd = (
                        now - closed_start
                        if closed_start is not None
                        else 0.0
                    )

                    draw_text(
                        display,
                        f"Both closed: {cd:.2f}s",
                        (
                            panel_x + 15,
                            sy + 60
                        ),
                        0.37,
                        (220, 220, 220),
                        1
                    )

                    draw_text(
                        display,
                        f"Blink count: {blink_count}",
                        (
                            panel_x + 15,
                            sy + 82
                        ),
                        0.37,
                        (220, 220, 220),
                        1
                    )

                    ac = (
                        (0, 0, 255)
                        if alarm_manager.alarm_active
                        else
                        (140, 140, 140)
                    )

                    draw_text(
                        display,
                        (
                            "ALARM: ON"
                            if alarm_manager.alarm_active
                            else
                            "ALARM: OFF"
                        ),
                        (
                            panel_x + 15,
                            sy + 108
                        ),
                        0.42,
                        ac,
                        2
                    )

                # =================================================
                # NO FACE HUD
                # =================================================

                else:

                    draw_text(
                        display,
                        "LEFT EYE  : NO FACE",
                        (
                            panel_x + 15,
                            y + 32
                        ),
                        0.36,
                        (0, 165, 255),
                        1
                    )

                    draw_text(
                        display,
                        "RIGHT EYE : NO FACE",
                        (
                            panel_x + 15,
                            y + 57
                        ),
                        0.36,
                        (0, 165, 255),
                        1
                    )

                # =================================================
                # DROWSINESS ALARM BAR
                # =================================================

                if alarm_manager.alarm_active:

                    cv2.rectangle(
                        display,
                        (
                            18,
                            height - 58
                        ),
                        (
                            panel_x - 18,
                            height - 28
                        ),
                        (0, 0, 180),
                        -1
                    )

                    draw_text(
                        display,
                        "DROWSINESS ALARM",
                        (
                            30,
                            height - 37
                        ),
                        0.38,
                        (255, 255, 255),
                        1
                    )

                # =================================================
                # FPS
                # =================================================

                fps = (
                    frame_count /
                    max(
                        0.001,
                        now - program_start
                    )
                )

                draw_text(
                    display,
                    f"FPS {fps:.1f}",
                    (
                        18,
                        height - 18
                    ),
                    0.38,
                    (180, 180, 180),
                    1
                )

                # =================================================
                # CONTROLS
                # =================================================

                draw_text(
                    display,
                    "R  recalibrate     Q  quit",
                    (
                        max(
                            20,
                            width - 205
                        ),
                        height - 18
                    ),
                    0.32,
                    (165, 165, 165),
                    1
                )

                # =================================================
                # SHOW
                # =================================================

                cv2.imshow(
                    WINDOW_NAME,
                    display
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                # Treat the window X button as a normal quit request too.
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        key = ord("q")
                except cv2.error:
                    key = ord("q")

                # =================================================
                # QUIT
                # =================================================

                if key in (ord("q"), ord("Q"), 27):

                    break

                # =================================================
                # RECALIBRATION
                # =================================================

                elif key == ord("r"):

                    alarm_manager.new_closure()

                    calibration = True

                    calibration_start = (
                        time.monotonic()
                    )

                    left_ears.clear()
                    right_ears.clear()
                    left_ears_2d.clear()
                    right_ears_2d.clear()

                    left_blinks.clear()
                    right_blinks.clear()

                    left_eye_widths_2d.clear()
                    right_eye_widths_2d.clear()
                    left_eye_widths_3d.clear()
                    right_eye_widths_3d.clear()

                    left_classifier = (
                        EyeClassifier()
                    )

                    right_classifier = (
                        EyeClassifier()
                    )

                    closed_start = None

                    previous_both_closed = False
                    left_eye_closed_start = None
                    right_eye_closed_start = None
                    previous_left_eye_closed = False
                    previous_right_eye_closed = False
                    previous_left_blink_event = False
                    previous_right_blink_event = False
                    last_blink_count_time = -999.0

                    blink_count = 0

                    state = "CALIBRATING"

                    print(
                        "Recalibration started. "
                        "Keep BOTH eyes OPEN."
                    )

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        alarm_manager.stop()

        camera.release()

        cv2.destroyAllWindows()

        print(
            "Camera stopped."
        )

        print(
            "Face Landmarker stopped."
        )

        print(
            "Test completed."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()