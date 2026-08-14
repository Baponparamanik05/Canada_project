import cv2
import time
import os
import statistics
import threading
import winsound
from collections import deque
import mediapipe as mp

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

CALIBRATION_SECONDS = 2.0
SMOOTH_FRAMES = 5

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

def eye_aspect_ratio(landmarks, indices):

    p = [landmarks[i] for i in indices]

    def dist(a, b):
        return (
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2
        ) ** 0.5

    h = dist(p[0], p[3])

    if h < 1e-6:
        return 0.0

    return (
        dist(p[1], p[5]) +
        dist(p[2], p[4])
    ) / (2.0 * h)


# ============================================================
# EYE CLASSIFIER
# ============================================================

class EyeClassifier:

    def __init__(self):

        self.base_ear = 0.25
        self.base_blink = 0.0

        self.ear_history = deque(
            maxlen=SMOOTH_FRAMES
        )

        self.blink_history = deque(
            maxlen=SMOOTH_FRAMES
        )

        self.closed_history = deque(
            maxlen=SMOOTH_FRAMES
        )

    def calibrate(self, ears, blinks):

        if ears:
            self.base_ear = statistics.median(ears)

        if blinks:
            self.base_blink = statistics.median(blinks)

    def update(self, ear, blink):

        # ----------------------------------------------------
        # Smooth EAR and blink values
        # ----------------------------------------------------

        self.ear_history.append(ear)
        self.blink_history.append(blink)

        smooth_ear = statistics.median(
            self.ear_history
        )

        smooth_blink = statistics.median(
            self.blink_history
        )

        # ----------------------------------------------------
        # Adaptive thresholds
        # ----------------------------------------------------

        # Lower threshold helps prevent looking left/right
        # from being interpreted as an eye closure.

        ear_threshold = max(
            0.095,
            self.base_ear * 0.55
        )

        blink_threshold = max(
            0.40,
            self.base_blink + 0.30
        )

        # ----------------------------------------------------
        # Blink detection
        # ----------------------------------------------------

        blink_closed = (
            smooth_blink >= blink_threshold
        )

        # ----------------------------------------------------
        # Very strong EAR closure
        # ----------------------------------------------------

        # EAR is only allowed to confirm closure when it is
        # extremely low.
        #
        # This prevents normal EAR changes caused by looking
        # left/right from being treated as CLOSED.

        very_low_ear = (
            smooth_ear <= ear_threshold * 0.78
        )

        # ----------------------------------------------------
        # Final eye state
        # ----------------------------------------------------

        if blink_closed:

            raw_closed = True

        elif (
            very_low_ear
            and smooth_blink >= 0.18
        ):

            raw_closed = True

        else:

            raw_closed = False

        # ----------------------------------------------------
        # Temporal smoothing
        # ----------------------------------------------------

        self.closed_history.append(
            raw_closed
        )

        votes = sum(
            self.closed_history
        )

        closed = (
            votes >=
            max(
                2,
                len(self.closed_history) - 1
            )
        )

        return {
            "closed": closed,
            "ear": smooth_ear,
            "blink": smooth_blink,
            "ear_threshold": ear_threshold,
            "blink_threshold": blink_threshold
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

    left_blinks = []
    right_blinks = []

    closed_start = None

    previous_both_closed = False

    blink_count = 0

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
        "SLEEPING EYE DETECTION V5"
    )

    print(
        "Keep BOTH eyes OPEN for 2 seconds."
    )

    print(
        "Both eyes closed > 0.8s = "
        "DROWSY + ALARM + NOTIFICATION"
    )

    print(
        "Both eyes closed > 2.0s = SLEEPING"
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

                    left_ear = (
                        eye_aspect_ratio(
                            landmarks,
                            LEFT_EYE
                        )
                    )

                    right_ear = (
                        eye_aspect_ratio(
                            landmarks,
                            RIGHT_EYE
                        )
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

                    # =================================================
                    # CALIBRATION
                    # =================================================

                    if calibration:

                        left_ears.append(
                            left_ear
                        )

                        right_ears.append(
                            right_ear
                        )

                        left_blinks.append(
                            left_blink
                        )

                        right_blinks.append(
                            right_blink
                        )

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
                                left_blinks
                            )

                            right_classifier.calibrate(
                                right_ears,
                                right_blinks
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
                                left_blink
                            )
                        )

                        right_data = (
                            right_classifier.update(
                                right_ear,
                                right_blink
                            )
                        )

                        # ------------------------------------------------
                        # BOTH EYES CLOSED
                        # ------------------------------------------------

                        both_closed = (
                            left_data["closed"]
                            and
                            right_data["closed"]
                        )

                        # ------------------------------------------------
                        # START CLOSURE
                        # ------------------------------------------------

                        if (
                            both_closed
                            and
                            not previous_both_closed
                        ):

                            closed_start = now

                            alarm_manager.new_closure()

                        # ------------------------------------------------
                        # CLOSURE TIMER
                        # ------------------------------------------------

                        closed_duration = (
                            now - closed_start
                            if
                            closed_start is not None
                            else
                            0.0
                        )

                        # ------------------------------------------------
                        # EYES OPEN AGAIN
                        # ------------------------------------------------

                        if (
                            not both_closed
                            and
                            previous_both_closed
                            and
                            closed_start is not None
                        ):

                            duration = (
                                now -
                                closed_start
                            )

                            if (
                                0.08
                                <= duration
                                <
                                DROWSY_SECONDS
                            ):

                                blink_count += 1

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

                            state = "NORMAL"

                            alarm_manager.new_closure()

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

                        left_color = (
                            (0, 0, 255)
                            if
                            left_data["closed"]
                            else
                            (0, 255, 0)
                        )

                        right_color = (
                            (0, 0, 255)
                            if
                            right_data["closed"]
                            else
                            (0, 255, 0)
                        )

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
                    "EAR + BLINK  |  FAST MODE",
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
                        "CLOSED"
                        if left_data["closed"]
                        else
                        "OPEN"
                    )

                    rs = (
                        "CLOSED"
                        if right_data["closed"]
                        else
                        "OPEN"
                    )

                    lc = (
                        (0, 0, 255)
                        if left_data["closed"]
                        else
                        (70, 230, 90)
                    )

                    rc = (
                        (0, 0, 255)
                        if right_data["closed"]
                        else
                        (70, 230, 90)
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
                            f"EAR thresholds "
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

                    # =================================================
                    # DROWSINESS STATUS
                    # =================================================

                    sy = y + 165

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

                # =================================================
                # QUIT
                # =================================================

                if key == ord("q"):

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

                    left_blinks.clear()
                    right_blinks.clear()

                    left_classifier = (
                        EyeClassifier()
                    )

                    right_classifier = (
                        EyeClassifier()
                    )

                    closed_start = None

                    previous_both_closed = False

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