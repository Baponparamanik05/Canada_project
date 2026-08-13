import time


class DrowsinessEngine:
    """
    Converts eye OPEN/CLOSED states into:
    NORMAL -> BLINK -> DROWSY -> SLEEPING
    """

    def __init__(
        self,
        drowsy_seconds=0.8,
        sleeping_seconds=2.0,
        recovery_seconds=0.3,
    ):
        self.drowsy_seconds = drowsy_seconds
        self.sleeping_seconds = sleeping_seconds
        self.recovery_seconds = recovery_seconds

        self.closed_start = None
        self.open_start = None

        self.state = "NORMAL"
        self.closed_duration = 0.0
        self.blink_count = 0

        self._blink_registered = False

    def update(self, eyes_closed):
        now = time.monotonic()

        # -----------------------------
        # EYES CLOSED
        # -----------------------------
        if eyes_closed:

            self.open_start = None

            if self.closed_start is None:
                self.closed_start = now
                self._blink_registered = False

            self.closed_duration = now - self.closed_start

            # Short eye closure = blink
            if self.closed_duration < self.drowsy_seconds:
                self.state = "BLINK"

            # Longer closure = drowsy
            elif self.closed_duration < self.sleeping_seconds:
                self.state = "DROWSY"

            # Long closure = sleeping
            else:
                self.state = "SLEEPING"

        # -----------------------------
        # EYES OPEN
        # -----------------------------
        else:

            # Count a blink only when
            # eyes were previously closed
            if self.closed_start is not None:

                duration = now - self.closed_start

                if 0.05 <= duration < self.drowsy_seconds:
                    self.blink_count += 1

                self.closed_start = None
                self.closed_duration = 0.0
                self.open_start = now

            # Small recovery period
            if self.open_start is not None:
                if now - self.open_start >= self.recovery_seconds:
                    self.state = "NORMAL"

        return self.get_status()

    def get_status(self):
        return {
            "state": self.state,
            "closed_duration": round(self.closed_duration, 2),
            "blink_count": self.blink_count,
        }

    def reset(self):
        self.closed_start = None
        self.open_start = None
        self.state = "NORMAL"
        self.closed_duration = 0.0
        self.blink_count = 0
        self._blink_registered = False