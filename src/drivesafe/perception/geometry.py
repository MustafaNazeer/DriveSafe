import numpy as np

def eye_aspect_ratio(points: list) -> float:
    ''' 
    Eye points are oriented something like this:

           p1      p2
      p0                p3
           p5      p4 
    
    So we accept a list() called points, in the form pX, where X is the point: points = [(p0.x, p0.y), ..., (p5.x, p5.y)]

    Then we calculate and return the Eye Aspect Ratio (EAR)
    '''

    # Convert points to a np.array to process calculations easier
    points = np.asarray(points, dtype=float)
    
    # Check points shape to be 6 points of the form (x, y)
    if points.shape != (6, 2):
        raise ValueError(f"Expected 6 points, in the form (x, y), instead received shape: {points.shape}")
    
    # Calculate the eye height and width
    height = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    width = np.linalg.norm(points[0] - points[3])

    # Avoid division by 0
    if width < 1e-9:
        return 0.0

    # EAR calculation
    return float(height / (2 * width))