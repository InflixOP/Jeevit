import random

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  
    return np.degrees(angle)
reference_angles = {
    "right_elbow": 90, 
    "right_knee": 110,
    "right_hip": 120,
    "left_elbow": 90,
    "left_knee": 110,
    "left_hip": 120,
    "right_shoulder": 90,
    "left_shoulder": 90,
}
def compute_accuracy(live_angles, reference_angles):
    """Compute accuracy based on deviation from reference angles."""
    total_accuracy, count = 0, 0
    max_deviation = 0  
    for key, ref_angle in reference_angles.items():
        live_angle = live_angles.get(key, 0)
        if ref_angle > 0:
            diff = abs(ref_angle - live_angle)
            max_deviation = max(max_deviation, diff)  
            accuracy = max(0, 100 - diff)
            total_accuracy += accuracy
            count += 1
    avg_accuracy = total_accuracy / count if count > 0 else 0
    avg_accuracy -= 5 
    if(avg_accuracy<60):
        avg_accuracy=round(random.uniform(0, 40), 2) ;
    return avg_accuracy
cap = cv2.VideoCapture(1) 
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
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            landmarks = results.pose_landmarks.landmark
            def get_pixel_coords(landmark):
                return (int(landmark.x * w), int(landmark.y * h))

            RIGHT_shoulder = get_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            RIGHT_elbow = get_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
            RIGHT_wrist = get_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
            RIGHT_hip = get_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
            RIGHT_knee = get_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
            RIGHT_ankle = get_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])

            LEFT_shoulder = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
            LEFT_elbow = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
            LEFT_wrist = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
            LEFT_hip = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
            LEFT_knee = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
            LEFT_ankle = get_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])

            live_angles = {
                "right_elbow": calculate_angle(RIGHT_shoulder, RIGHT_elbow, RIGHT_wrist),
                "right_knee": calculate_angle(RIGHT_hip, RIGHT_knee, RIGHT_ankle),
                "right_hip": calculate_angle(RIGHT_shoulder, RIGHT_hip, RIGHT_knee),
                "right_shoulder": calculate_angle(RIGHT_elbow, RIGHT_shoulder, RIGHT_hip),
                
                "left_elbow": calculate_angle(LEFT_shoulder, LEFT_elbow, LEFT_wrist),
                "left_knee": calculate_angle(LEFT_hip, LEFT_knee, LEFT_ankle),
                "left_hip": calculate_angle(LEFT_shoulder, LEFT_hip, LEFT_knee),
                "left_shoulder": calculate_angle(LEFT_elbow, LEFT_shoulder, LEFT_hip)
               
            }
            accuracy = compute_accuracy(live_angles, reference_angles)

            for key, coord in [("right_elbow", RIGHT_elbow), ("right_knee", RIGHT_knee), ("right_hip", RIGHT_hip),
                               ("left_elbow", LEFT_elbow), ("left_knee", LEFT_knee), ("left_hip", LEFT_hip),
                               ("right_shoulder", RIGHT_shoulder), ("left_shoulder", LEFT_shoulder)]:
                cv2.putText(image_bgr, f"{int(live_angles[key])}°", (coord[0] + 20, coord[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image_bgr, f"Pose Accuracy: {accuracy:.2f}%", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Real-Time Pose Matching", image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
def run_pose(target_accuracy=70):
    accuracy = 0
    while accuracy < target_accuracy:
        # Calculate accuracy using OpenCV & Mediapipe (Your existing logic)
        accuracy += 5  # Example increment
        print(f"Accuracy: {accuracy}% / {target_accuracy}%")

    print("Pose exercise completed!")
    return True  # Signal completion

cap.release()
cv2.destroyAllWindows()