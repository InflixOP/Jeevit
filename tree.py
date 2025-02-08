import time

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
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid numerical errors
    return np.degrees(angle)

reference_angles = {
    "elbow": 48, 
    "knee": 45,
    "hip": 106
}

def compute_accuracy(live_angles, reference_angles):
    total_accuracy = 0
    count = 0
    max_deviation = 0  

    for key in reference_angles:
        ref_angle = reference_angles[key]
        live_angle = live_angles.get(key, 0)

        if ref_angle > 0:
            diff = abs(ref_angle - live_angle)
            max_deviation = max(max_deviation, diff)  

            accuracy = max(0, 100 - diff)  
            total_accuracy += accuracy
            count += 1
    avg_accuracy = total_accuracy / count if count > 0 else 0
    avg_accuracy -= 5

    if max_deviation > 30:  
        return min(avg_accuracy, 38) 
    elif max_deviation > 20:  
        return avg_accuracy * 0.7 
    
    return avg_accuracy

cap = cv2.VideoCapture(1)  

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    success_time = None
    target_duration = 3
    target_accuracy = 70
    
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

            live_angles = {
                "elbow": calculate_angle(RIGHT_shoulder, RIGHT_elbow, RIGHT_wrist),
                "knee": calculate_angle(RIGHT_hip, RIGHT_knee, RIGHT_ankle),
                "hip": calculate_angle(RIGHT_shoulder, RIGHT_hip, RIGHT_knee),
            }

            accuracy = compute_accuracy(live_angles, reference_angles)

            if accuracy >= target_accuracy:
                if success_time is None:
                    success_time = time.time()
                elif time.time() - success_time >= target_duration:
                    cv2.putText(image_bgr, "Exercise Completed!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.imshow("Real-Time Pose Matching", image_bgr)
                    cv2.waitKey(3000)  
                    break
            else:
                success_time = None  

            cv2.putText(image_bgr, f"{int(live_angles['elbow'])}°", (RIGHT_elbow[0] + 20, RIGHT_elbow[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image_bgr, f"{int(live_angles['knee'])}°", (RIGHT_knee[0] + 20, RIGHT_knee[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image_bgr, f"{int(live_angles['hip'])}°", (RIGHT_hip[0] + 20, RIGHT_hip[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image_bgr, f"Pose Accuracy: {accuracy:.2f}%", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Real-Time Pose Matching", image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()