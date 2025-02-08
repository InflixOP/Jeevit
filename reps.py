import random
import time

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

reference_angles = {"elbow": 48, "knee": 45, "hip": 106}

def compute_accuracy(live_angles, reference_angles):
    total_accuracy, count= 0, 0
    for key in reference_angles:
        ref_angle = reference_angles[key]
        live_angle = live_angles.get(key, 0)
        if ref_angle > 0:
            diff = abs(ref_angle - live_angle)
            accuracy = max(0, 100 - diff)
            total_accuracy += accuracy
            count += 1
    avg_accuracy = total_accuracy / count if count > 0 else 0
    avg_accuracy -= 5  
    if(avg_accuracy<60):
        avg_accuracy=round(random.uniform(0, 40), 2) ;
    return avg_accuracy

cap = cv2.VideoCapture(1)  

reps, rep_started = 0, False
pose_accuracy_phase = False
start_time, pose_held_time = None, 4  

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

            elbow_angle = calculate_angle(RIGHT_shoulder, RIGHT_elbow, RIGHT_wrist)
            knee_angle = calculate_angle(RIGHT_hip, RIGHT_knee, RIGHT_ankle)
            hip_angle = calculate_angle(RIGHT_shoulder, RIGHT_hip, RIGHT_knee)

            live_angles = {"elbow": elbow_angle, "knee": knee_angle, "hip": hip_angle}

            if not pose_accuracy_phase:
                if elbow_angle < 50 and not rep_started:
                    rep_started = True  
                elif elbow_angle > 160 and rep_started:
                    reps += 1
                    rep_started = False

                cv2.putText(image_bgr, f"Reps: {reps}/10", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                if reps >= 3:
                    pose_accuracy_phase = True
                    start_time = None  
            
            else:
                accuracy = compute_accuracy(live_angles, reference_angles)

                if accuracy >= 70:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= pose_held_time:
                        break  
                else:
                    start_time = None  

                cv2.putText(image_bgr, f"Pose Accuracy: {accuracy:.2f}%", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Exercise", image_bgr)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("🎯 Exercise Completed! 🎯")
