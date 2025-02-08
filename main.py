import random
import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("yoga_pose_model.h5")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
timer=0
reference_angles1 = {"elbow": 48, "knee": 45, "hip": 106}
reference_angles2 = {
    "right_elbow": 180, 
    "right_knee": 90,
    "right_hip": 120,
    "left_elbow": 180,
    "left_knee": 160,
    "left_hip": 150,
    "right_shoulder": 90,
    "left_shoulder": 90,
    "right_ankle": 90, 
    "left_ankle": 90
}
reference_angles3 = {
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
    total_accuracy, count, max_deviation = 0, 0, 0
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
    if(avg_accuracy<60):
        avg_accuracy=round(random.uniform(0, 40), 2) ;
    return avg_accuracy

cap = cv2.VideoCapture(1)  
rimage=cv2.imread("reps.jpg")
left_reps, right_reps, target_reps = 0, 0, 1
left_curl, right_curl = False, False
reps=True
pose_accuracy_phase = False
pose_accuracy_phase1 = False
pose_accuracy_phase2 = False
pose_accuracy_phase3 = False
pose_accuracy_phase4=False
start_time, pose_held_time = None, 3  
overlay_width, overlay_height = 150, 100  
rimage = cv2.resize(rimage, (overlay_width, overlay_height))
timage=cv2.imread("C:\\Users\\saxen\\Jeevit\\yoga\\TEST\\tree\\00000000.jpg")
wimage=cv2.imread("C:\\Users\\saxen\\Jeevit\\yoga\\TEST\\warrior2\\00000001.jpg")
gimage=cv2.imread("C:\\Users\\saxen\\Jeevit\\yoga\\TEST\\goddess\\00000000.jpg")
dimage=cv2.imread("C:\\Users\\saxen\\Jeevit\\yoga\\TEST\\downdog\\00000000.jpg")
pimage=cv2.imread("C:\\Users\\saxen\\Jeevit\\yoga\\TEST\\plank\\00000021.jpg")
timage = cv2.resize(timage, (overlay_width, overlay_height))
wimage = cv2.resize(wimage, (overlay_width, overlay_height))
gimage = cv2.resize(gimage, (overlay_width, overlay_height))
dimage = cv2.resize(dimage, (overlay_width, overlay_height))
pimage = cv2.resize(pimage, (overlay_width, overlay_height))
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
        h, w, _ = image_bgr.shape
        original_bgr = image_bgr.copy()
        x_offset = w - overlay_width   
        y_offset =  0 
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

            left_elbow_angle = calculate_angle(LEFT_shoulder, LEFT_elbow, LEFT_wrist)
            right_elbow_angle = calculate_angle(RIGHT_shoulder, RIGHT_elbow, RIGHT_wrist)
            knee_angle = calculate_angle(RIGHT_hip, RIGHT_knee, RIGHT_ankle)
            hip_angle = calculate_angle(RIGHT_shoulder, RIGHT_hip, RIGHT_knee)

            live_angles1 = {
                "elbow": calculate_angle(RIGHT_shoulder, RIGHT_elbow, RIGHT_wrist),
                "knee": calculate_angle(RIGHT_hip, RIGHT_knee, RIGHT_ankle),
                "hip": calculate_angle(RIGHT_shoulder, RIGHT_hip, RIGHT_knee),
            }
            live_angles2 = {
                "right_elbow": calculate_angle(RIGHT_shoulder, RIGHT_elbow, RIGHT_wrist),
                "right_knee": calculate_angle(RIGHT_hip, RIGHT_knee, RIGHT_ankle),
                "right_hip": calculate_angle(RIGHT_shoulder, RIGHT_hip, RIGHT_knee),
                "right_shoulder": calculate_angle(RIGHT_elbow, RIGHT_shoulder, RIGHT_hip),
                "right_ankle": calculate_angle(RIGHT_knee, RIGHT_ankle, RIGHT_hip),

                "left_elbow": calculate_angle(LEFT_shoulder, LEFT_elbow, LEFT_wrist),
                "left_knee": calculate_angle(LEFT_hip, LEFT_knee, LEFT_ankle),
                "left_hip": calculate_angle(LEFT_shoulder, LEFT_hip, LEFT_knee),
                "left_shoulder": calculate_angle(LEFT_elbow, LEFT_shoulder, LEFT_hip),
                "left_ankle": calculate_angle(LEFT_knee, LEFT_ankle, LEFT_hip),
            }
            live_angles3 = {
                "right_elbow": calculate_angle(RIGHT_shoulder, RIGHT_elbow, RIGHT_wrist),
                "right_knee": calculate_angle(RIGHT_hip, RIGHT_knee, RIGHT_ankle),
                "right_hip": calculate_angle(RIGHT_shoulder, RIGHT_hip, RIGHT_knee),
                "right_shoulder": calculate_angle(RIGHT_elbow, RIGHT_shoulder, RIGHT_hip),
                
                "left_elbow": calculate_angle(LEFT_shoulder, LEFT_elbow, LEFT_wrist),
                "left_knee": calculate_angle(LEFT_hip, LEFT_knee, LEFT_ankle),
                "left_hip": calculate_angle(LEFT_shoulder, LEFT_hip, LEFT_knee),
                "left_shoulder": calculate_angle(LEFT_elbow, LEFT_shoulder, LEFT_hip)
               
            }
            
            if reps :
                image_bgr[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = rimage
                if left_elbow_angle < 40:
                    left_curl = True
                if left_curl and left_elbow_angle > 160:
                    left_reps += 1
                    left_curl = False

                if right_elbow_angle < 40:
                    right_curl = True
                if right_curl and right_elbow_angle > 160:
                    right_reps += 1
                    right_curl = False

                cv2.putText(image_bgr, f"{int(left_elbow_angle)}°", (LEFT_elbow[0] + 20, LEFT_elbow[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"{int(right_elbow_angle)}°", (RIGHT_elbow[0] + 20, RIGHT_elbow[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image_bgr, f"Left Reps: {left_reps}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Right Reps: {right_reps}", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if left_reps >= target_reps or right_reps >= target_reps:
                        image_bgr = np.zeros((500, 800, 3), dtype=np.uint8)  
                        cv2.putText(image_bgr, "Exercise Completed!", (100, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow("Exercise", image_bgr)  
                        cv2.waitKey(3000)  
                        reps=False
                        pose_accuracy_phase = True
                        start_time = None  
                        image_bgr = np.zeros((500, 800, 3), dtype=np.uint8)

            elif pose_accuracy_phase:
                image_bgr[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = timage
                accuracy = compute_accuracy(live_angles1, reference_angles1)
                if accuracy >= 70:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= pose_held_time:
                        pose_accuracy_phase=False
                        pose_accuracy_phase1=True
                        start_time=None
                        image_bgr = np.zeros((500, 800, 3), dtype=np.uint8)  
                        cv2.putText(image_bgr, "Exercise Completed!", (100, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow("Exercise", image_bgr)  
                        cv2.waitKey(3000)  
                else:
                    start_time = None  

                cv2.putText(image_bgr, f"Pose Accuracy: {accuracy:.2f}%", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
            elif pose_accuracy_phase1:
                image_bgr[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = wimage
                accuracy = compute_accuracy(live_angles2, reference_angles2)

                if accuracy >= 70:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= pose_held_time:
                        pose_accuracy_phase1=False
                        pose_accuracy_phase2=True
                        start_time=None
                        image_bgr = np.zeros((500, 800, 3), dtype=np.uint8)  
                        cv2.putText(image_bgr, "Exercise Completed!", (100, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow("Exercise", image_bgr)  
                        cv2.waitKey(3000)  
                else:
                    start_time = None  

                cv2.putText(image_bgr, f"Pose Accuracy: {accuracy:.2f}%", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif pose_accuracy_phase2:
                image_bgr[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = gimage
                accuracy = compute_accuracy(live_angles3, reference_angles3)

                if accuracy >= 70:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= pose_held_time:
                        image_bgr = np.zeros((500, 800, 3), dtype=np.uint8)  
                        cv2.putText(image_bgr, "Exercise Completed!", (100, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow("Exercise", image_bgr)  
                        cv2.waitKey(3000)  
                        pose_accuracy_phase2=False
                        pose_accuracy_phase3=True
                        start_time=None
                else:
                    start_time = None  

                cv2.putText(image_bgr, f"Pose Accuracy: {accuracy:.2f}%", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif pose_accuracy_phase3:
                image_bgr[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = dimage
                if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility:
                    side = "Left"
                    reference_angles = {"left_elbow": 150  , "left_knee": 180, "left_hip": 90, "left_shoulder": 180}
                    live_angles = {
                        "left_elbow": calculate_angle(LEFT_shoulder, LEFT_elbow, LEFT_wrist),
                        "left_knee": calculate_angle(LEFT_hip, LEFT_knee, LEFT_ankle),
                        "left_hip": calculate_angle(LEFT_shoulder, LEFT_hip, LEFT_knee),
                        "left_shoulder": calculate_angle(LEFT_elbow, LEFT_shoulder, LEFT_hip)
                     }
                    keypoints = [("left_elbow", LEFT_elbow), ("left_knee", LEFT_knee),
                             ("left_hip", LEFT_hip), ("left_shoulder", LEFT_shoulder)]
                else:
                    side = "Right"
                    reference_angles = {"right_elbow": 150, "right_knee": 180, "right_hip": 90, "right_shoulder": 180}
                    live_angles = {
                        "right_elbow": calculate_angle(RIGHT_shoulder, RIGHT_elbow, RIGHT_wrist),
                        "right_knee": calculate_angle(RIGHT_hip, RIGHT_knee, RIGHT_ankle),
                        "right_hip": calculate_angle(RIGHT_shoulder, RIGHT_hip, RIGHT_knee),
                        "right_shoulder": calculate_angle(RIGHT_elbow, RIGHT_shoulder, RIGHT_hip)
                     }
                    keypoints = [("right_elbow", RIGHT_elbow), ("right_knee", RIGHT_knee),
                             ("right_hip", RIGHT_hip), ("right_shoulder", RIGHT_shoulder)]

                accuracy = compute_accuracy(live_angles, reference_angles)
                if accuracy >= 70:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= pose_held_time:
                        image_bgr = np.zeros((500, 800, 3), dtype=np.uint8)  
                        cv2.putText(image_bgr, "Exercise Completed!", (100, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow("Exercise", image_bgr)  
                        cv2.waitKey(3000)  
                        pose_accuracy_phase3=False
                        pose_accuracy_phase4=True
                        start_time=None
                else:
                    start_time = None  

                cv2.putText(image_bgr, f"Pose Accuracy: {accuracy:.2f}%", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                image_bgr[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = pimage
                if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility:
                    reference_angles = {"left_elbow": 90, "left_knee": 180, "left_hip": 160, "left_shoulder": 90}
                    live_angles = {
                        "left_elbow": calculate_angle(LEFT_shoulder, LEFT_elbow, LEFT_wrist),
                        "left_knee": calculate_angle(LEFT_hip, LEFT_knee, LEFT_ankle),
                        "left_hip": calculate_angle(LEFT_shoulder, LEFT_hip, LEFT_knee),
                        "left_shoulder": calculate_angle(LEFT_elbow, LEFT_shoulder, LEFT_hip)
                     }
                    keypoints = [("left_elbow", LEFT_elbow), ("left_knee", LEFT_knee),
                             ("left_hip", LEFT_hip), ("left_shoulder", LEFT_shoulder)]
                else:
                    side = "Right"
                    reference_angles = {"right_elbow": 90, "right_knee": 180, "right_hip": 160, "right_shoulder": 90}
                    live_angles = {
                        "right_elbow": calculate_angle(RIGHT_shoulder, RIGHT_elbow, RIGHT_wrist),
                        "right_knee": calculate_angle(RIGHT_hip, RIGHT_knee, RIGHT_ankle),
                        "right_hip": calculate_angle(RIGHT_shoulder, RIGHT_hip, RIGHT_knee),
                        "right_shoulder": calculate_angle(RIGHT_elbow, RIGHT_shoulder, RIGHT_hip)
                        }
                    keypoints = [("right_elbow", RIGHT_elbow), ("right_knee", RIGHT_knee),
                             ("right_hip", RIGHT_hip), ("right_shoulder", RIGHT_shoulder)]

                accuracy = compute_accuracy(live_angles, reference_angles)
                if accuracy >= 70:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= pose_held_time:
                        image_bgr = np.zeros((500, 800, 3), dtype=np.uint8)  
                        cv2.putText(image_bgr, "All Exercises Completed!", (100, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow("Exercise", image_bgr)  
                        cv2.waitKey(3000)  
                        pose_accuracy_phase4=False
                        start_time=None
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
