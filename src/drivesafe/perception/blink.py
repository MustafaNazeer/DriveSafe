class BlinkDetector:

    def __init__(self, threshold=0.21, consecutive_frames=2, closure_frames=30):
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.closure_frames = closure_frames
        self.alarm = False

        self.blink_count = 0
        self.closed_frames = 0

    def is_closed(self, ear):
        return ear <= self.threshold
    
    # Checks if the eye is closed, increments the closed_frames if so, otherwise
    # checks to see if the closed frames is at least consecutive frames, increments blinks if so,
    # then finally resets the closed frames back to 0, since the blink is done (eye has re-opened)
    def update(self, ear) -> None:

        if self.is_closed(ear):
            self.closed_frames += 1
        else:
            if self.closed_frames >= self.consecutive_frames:
                self.blink_count += 1
            self.closed_frames = 0
        
        self.alarm = self.closed_frames >= self.closure_frames