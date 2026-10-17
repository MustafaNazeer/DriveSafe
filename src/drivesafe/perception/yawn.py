class YawnDetector:
    """
    A yawn is MAR staying above a threshold for a minimum time.

    The deafults come from YawDD: tuned on the female mirror videos, scored once on the male ones. 
    35 frames at YawDD's ~30 fps is ~1.2 s, so the duration is kept in seconds
    because the demo runs at a different frame rate than the dataset 
    """

    def __init__(self, threshold=0.35, min_duration=1.2):
        self.threshold = threshold
        self.min_duration = min_duration

        self.yawn_count = 0
        self.yawning = False
        self.above_since = None
        self.counted = False

    def update(self, mar, now) -> None:
        if mar > self.threshold:
            if self.above_since is None:
                self.above_since = now

            if now - self.above_since >= self.min_duration:
                self.yawning = True
                #count each yawn once, not once per frame
                if not self.counted:
                    self.yawn_count += 1
                    self.counted = True
        else:
            self.reset()

    def reset(self) -> None:
        #Also called when the face is lost, since a lost face breaks the run
        self.above_since = None
        self.yawning = False
        self.counted = False