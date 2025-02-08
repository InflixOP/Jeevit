import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained yoga pose classification model
model = tf.keras.models.load_model("yoga_pose_model.h5")

# Define class labels (Update these based on your dataset)
class_labels = ["downdog", "goddess", "plank", "tree", "warrior2"]

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Start video capture
cap = cv2.VideoCapture(1)  # Change to 0 if using a built-in webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Detect pose landmarks
        results = pose.process(image)
        
        # Convert image back to BGR for OpenCV rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )

        # ✅ Preprocess the frame for the model (resize & normalize)
        img = cv2.resize(frame, (224, 224))  # Resize to model's input size
        img = img.astype("float32") / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # ✅ Make prediction
        predictions = model.predict(img)
        predicted_index = np.argmax(predictions)  # Get predicted class index
        confidence = np.max(predictions)  # Get confidence score

        # ✅ Display prediction on the frame
        label_text = f"{class_labels[predicted_index]} ({confidence*100:.2f}%)"
        cv2.putText(image, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the output frame
        cv2.imshow("Yoga Pose Recognition", image)

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
