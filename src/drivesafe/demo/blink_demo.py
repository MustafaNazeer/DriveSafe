from pathlib import Path
import cv2
import mediapipe as mp
import argparse
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from time import monotonic

from drivesafe.perception.eye_state import EyeStateClassifier

from drivesafe.perception.blink import BlinkDetector
from drivesafe.perception.perclos import Perclos
from drivesafe.perception.landmarks import to_pixel_array, average_eye_aspect_ratio, extract_points, LEFT_EYE_INDICES, RIGHT_EYE_INDICES

# Eye aspect ratio baselines used to convert an EAR reading into a percent
# closed for PERCLOS. Measured across two runs on one subject (Mustafa) with
# his webcam under indoor lighting. The median EAR with the eyes open, and the
# median with them shut. Together they put the P80 closure cutoff at EAR 0.0768.
#
# Specific to one face, one camera, one lighting condition. Re-measure if any of
# those change, and update the Results section of
# src/drivesafe/perception/perclos-calibration.md to match.
EAR_OPEN = 0.240
EAR_CLOSED = 0.036
WINDOW_SIZE = 15.0
WINDOW_TOLERANCE = 0.5

# Path to MediaPipe model file on disk
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"

#Path to the Model
CNN_MODEL_PATH = REPO_ROOT / "models" / "eye_state_cnn.pt"


# Parser is used to determine whether to run the program on rpi5 or laptop webcam by argument
def parse_args():
    parser = argparse.ArgumentParser(
        description = "DriveSafe blink and drowsiness demo"
    )
    parser.add_argument(
        "--camera",
        choices=["webcam","pi"],
        default = "webcam",
        help="Camera source: webcam for laptop/USB camera, pi for Raspberry Pi Camera Module",
    )
    return parser.parse_args()


def main():
    # Sets argument parser
    args = parse_args()

    # Checks to see if face_landmarker.task is on disk
    if not MODEL_PATH.exists():
        print(f"Face landmarker model not found at {MODEL_PATH}")
        return

    # Load MODEL_PATH file, track the state between frames via VIDEO mode, and tell VIDEO how many faces to look for
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    # Read .task file, unpack options model, and create MediaPipe's interface graph
    landmarker = vision.FaceLandmarker.create_from_options(options)
    
    # Open camera
    cap = None
    picam2 = None

    if args.camera == "webcam":
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("No webcam found.")
            return

        print("Using webcam")
    
    elif args.camera == "pi":
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("Picamera2 is not installed or available.")
            return

        picam2 = Picamera2()
    
        camera_config = picam2.create_preview_configuration(
            main={"size":(640, 480), "format": "RGB888"}
        )

        picam2.configure(camera_config)
        picam2.start()

        print("Using Raspberry Pi Camera Module")
    
    # Initilize a BlinkDetector to check if eye is closed, track blinks, and update closed frames / eye state
    detector = BlinkDetector()
    #Initialize a Perclos to implement a window of the last minute in the demo
    perclos = Perclos(WINDOW_SIZE, EAR_OPEN, EAR_CLOSED)

    # Load the trained eye state CNN 
    eye_state_classifier = EyeStateClassifier(CNN_MODEL_PATH)

    # Write down starting time of demo
    start = monotonic()

    # Frame loop to iterate for every frame
    while True:

        # Write down current time (seconds)
        now = monotonic()

        # Obtain frame, then check to see if the camera is still on for this frame
        if args.camera == "webcam":
            ok, frame = cap.read()
  
            if not ok:
                break
            
            # Webcam/OpenCV provides BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        elif args.camera == "pi":
            #With our tested Pi Camera configuration, this array displays correctly thourgh OpenCV 
            frame = picam2.capture_array()

            # Mediapipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize the MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Obtain height and weight of eye
        h, w = frame.shape[:2]

        # Analyze the face on the camera, then label with the timestamp value
        result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))


        # Check to see if a face was analyzed / detected
        if not result.face_landmarks:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:

            # Obtain 6 landmarks, calculate ear, then update state of eye (closed and closure frames)
            landmarks = to_pixel_array(result.face_landmarks[0], w, h)
            ear = average_eye_aspect_ratio(landmarks)
            detector.update(ear)
            perclos.update(ear, now)

            # Draw the 6 points on the left and right eyes of the face
            for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
                for x, y in extract_points(landmarks, indices):
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            #CNN eye state classification, using the same landmarks EAR is computed from
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cnn_label = eye_state_classifier.predict(gray, landmarks)

            # Print EAR, blink count, eye closure percentage, and window fill to camera
            cv2.putText(frame, f"EAR: {ear:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {detector.blink_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Closure Percentage: {perclos.perclos() * 100:.1f}%", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Eye State: {cnn_label}", (20, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            if perclos.window_fill() < perclos.window - WINDOW_TOLERANCE:
                cv2.putText(frame, f"Window Fill: {perclos.window_fill():.1f} seconds", (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Fire drowsy alarm if BlinkDetector set the detector alarm
            if detector.alarm:
                cv2.putText(frame, "DROWSY", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Display frame to screen
        cv2.imshow("DriveSafe EAR Demo", frame)

        # End frame loop (close program), if user entered 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close camera, then close camera display window
    if cap is not None:
        cap.release()
    if picam2 is not None:
        picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
