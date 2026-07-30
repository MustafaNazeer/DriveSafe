from collections import deque

# PERCLOS is the percentage of time within a window that the eyelids cover at least 80 percent of the pupil
P80_CLOSURE_FRACTION = 0.80

class Perclos:

    def __init__(self, window=60.0, earOpen=None, earClosed=None):
        self.window = window
        self.earOpen = None
        self.earClosed = None
        self.frames = deque()

        if earOpen is not None and earClosed is not None:
            self.calibrate(earOpen, earClosed)
    
    # Seperate function from __init__ just to validate on passed in earOpen and passed in earClosed
    def calibrate(self, earOpen, earClosed):
        if earOpen <= earClosed:
            raise ValueError(f"EAR open value must be larger than that of EAR closed: Passed in EAR open: {earOpen}. Passed in EAR closed: {earClosed}.")
        
        self.earOpen = earOpen
        self.earClosed = earClosed
    
    def is_calibrated(self):
        return self.earOpen is not None and self.earClosed is not None

    # Calculate how closed the eye is within a range of [0, 1], 0 = fully closed, 1 = fully open
    def closed_ratio(self, ear):
        if self.is_calibrated():
            ratio = (self.earOpen - ear) / (self.earOpen - self.earClosed)
            ratio = max(0.0, min(1.0, ratio))
            return ratio
        raise ValueError("EAR open and EAR closed values were never calibrated.")

    # Update the frames deque for the next frame, and evict any frames that are too old
    def update(self, ear, t):
        if not self.is_calibrated():
            return
        
        closed_ratio = self.closed_ratio(ear)
        is_closed = closed_ratio >= P80_CLOSURE_FRACTION

        # Record closed ratio of the eye, if its closed, and current timestamp
        self.frames.append((closed_ratio, is_closed, t))

        # Evict if timestamp is too old (out of window)
        while self.frames and self.frames[0][2] < (t - self.window):
            self.frames.popleft()

    # Return PERCLOS ratio of the entire window
    def perclos(self):

        if not self.is_calibrated() or not self.frames:
            return None
        
        total_closed = 0
        for _, is_closed, _ in self.frames:
            if is_closed:
                total_closed += 1
        return total_closed / len(self.frames)

    # Return size of the window calculated in timestamps
    def window_fill(self):
        return self.frames[-1][2] - self.frames[0][2] if self.frames else 0.0