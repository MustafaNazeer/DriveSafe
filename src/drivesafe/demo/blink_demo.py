from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from drivesafe.perception.blink import BlinkDetector
from drivesafe.perception.landmarks import to_pixel_array, average_eye_aspect_ratio, extract_points, LEFT_EYE_INDICES, RIGHT_EYE_INDICES

# Path to MediaPipe model file on disk
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"

def main():
    
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
    cap = cv2.VideoCapture(0)

    # Check if camera opened / is found
    if not cap.isOpened():
        print("No webcam found.")
        return
    
    # Initilize a BlinkDetector to check if eye is closed, track blinks, and update closed frames / eye state
    detector = BlinkDetector()


    frame_index = 0

    # Frame loop to iterate for every frame
    while True:

        # Obtain frame, then check to see if the camera is still on for this frame
        ok, frame = cap.read()
        if not ok:
            break

        # Convert to RGB color scheme
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize the MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Obtain height and weight of eye
        h, w = frame.shape[:2]

        # Capture the timestamp, in ms, of the frame
        timestamp_ms = int(frame_index * 1000 / 20)

        # Analyze the face on the camera, then label with the timestamp value
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Iterate for next frame
        frame_index += 1


        # Check to see if a face was analyzed / detected
        if not result.face_landmarks:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:

            # Obtain 6 landmarks, calculate ear, then update state of eye (closed and closure frames)
            landmarks = to_pixel_array(result.face_landmarks[0], w, h)
            ear = average_eye_aspect_ratio(landmarks)
            detector.update(ear)

            # Draw the 6 points on the left and right eyes of the face
            for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
                for x, y in extract_points(landmarks, indices):
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            # Print EAR and blink count to camera
            cv2.putText(frame, f"EAR: {ear:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {detector.blink_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Fire drowsy alarm if BlinkDetector set the detector alarm
            if detector.alarm:
                cv2.putText(frame, "DROWSY", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Display frame to screen
        cv2.imshow("DriveSafe EAR Demo", frame)

        # End frame loop (close program), if user entered 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close camera, then close camera display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()