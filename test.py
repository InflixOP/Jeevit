import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Open webcam
cap = cv2.VideoCapture(1)

# Rep counting variables
left_reps, right_reps = 0, 0
left_curl, right_curl = False, False

# Start Pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            landmarks = results.pose_landmarks.landmark

            def get_pixel_coords(landmark):
                return (int(landmark.x * w), int(landmark.y * h))

            # Get coordinates for both arms
            LEFT_shoulder = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
            LEFT_elbow = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
            LEFT_wrist = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])

            RIGHT_shoulder = get_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            RIGHT_elbow = get_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
            RIGHT_wrist = get_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])

            # Calculate elbow angles
            left_elbow_angle = calculate_angle(LEFT_shoulder, LEFT_elbow, LEFT_wrist)
            right_elbow_angle = calculate_angle(RIGHT_shoulder, RIGHT_elbow, RIGHT_wrist)

            # Rep counting logic
            # LEFT ARM
            if left_elbow_angle < 40:  # Arm is flexed (up position)
                left_curl = True
            if left_curl and left_elbow_angle > 160:  # Arm extended (down position)
                left_reps += 1
                left_curl = False

            # RIGHT ARM
            if right_elbow_angle < 40:  
                right_curl = True
            if right_curl and right_elbow_angle > 160:  
                right_reps += 1
                right_curl = False

            # Display angles
            cv2.putText(image_bgr, f"{int(left_elbow_angle)}°", (LEFT_elbow[0] + 20, LEFT_elbow[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_bgr, f"{int(right_elbow_angle)}°", (RIGHT_elbow[0] + 20, RIGHT_elbow[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Display rep count on screen
        cv2.putText(image_bgr, f"Left Reps: {left_reps}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image_bgr, f"Right Reps: {right_reps}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Dumbbell Curl Rep Counter", image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
