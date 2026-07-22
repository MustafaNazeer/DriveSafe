from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from drivesafe.perception.blink import BlinkDetector
from drivesafe.perception.landmarks import to_pixel_array, average_eye_aspect_ratio, extract_points, LEFT_EYE_INDICES, RIGHT_EYE_INDICES

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"

def main():
    if not MODEL_PATH.exists():
        print(f"Face landmarker model not found at {MODEL_PATH}")
        return



    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No webcam found.")
        return
    
    detector = BlinkDetector()
    frame_index = 0

    while True:

        ok, frame = cap.read()
        if not ok:
            break


        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        
        h, w = frame.shape[:2]
        timestamp_ms = int(frame_index * 1000 / 20)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_index += 1

        if not result.face_landmarks:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            landmarks = to_pixel_array(result.face_landmarks[0], w, h)
            ear = average_eye_aspect_ratio(landmarks)
            detector.update(ear)

            for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
                for x, y in extract_points(landmarks, indices):
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            
            cv2.putText(frame, f"EAR: {ear:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {detector.blink_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


            if detector.alarm:
                cv2.putText(frame, "DROWSY", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)


        cv2.imshow("DriveSafe EAR Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()