import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import LSTM

# Define a custom LSTM that ignores the 'time_major' argument
class MyLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        # Remove 'time_major' if it exists
        kwargs.pop('time_major', None)
        super(MyLSTM, self).__init__(*args, **kwargs)

# Load the trained model using the custom LSTM layer
model = tf.keras.models.load_model("action.h5", custom_objects={'LSTM': MyLSTM})

# Define class labels (Make sure these match your model's classes!)
actions = np.array(['hello', 'thanks', 'iloveyou'])  # Modify if needed

# Initialize MediaPipe models
mp_holistic = mp.solutions.holistic  # Full-body detection model
mp_drawing = mp.solutions.drawing_utils  # Utility to draw landmarks
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

print("[INFO] Initializing webcam...")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

if not cap.isOpened():
    print("[ERROR] Could not open webcam. Check camera permissions or close other apps using the webcam.")
    exit()

print("[INFO] Webcam initialized successfully!")

# Function to detect and process the frame
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    
    return np.concatenate([pose, face, lh, rh])

# Function to draw landmarks on frame
def draw_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

print("[INFO] Loading MediaPipe model...")

# Initialize prediction variables
sequence = []  # Stores the last few frames
sequence_length = 30  # How many frames to analyze

try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print("[INFO] MediaPipe model initialized successfully!")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Failed to capture frame.")
                break

            # Process frame with MediaPipe
            image, results = mediapipe_detection(frame, holistic)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Keep only the last `sequence_length` frames
            sequence = sequence[-sequence_length:]

            # Predict only if we have enough frames
            if len(sequence) == sequence_length:
                input_data = np.expand_dims(sequence, axis=0)  # Shape (1, 30, 1662)
                prediction = model.predict(input_data)[0]  # Get predictions
                predicted_label = actions[np.argmax(prediction)]  # Get the most likely action

                print(f"[PREDICTION] {predicted_label} (Confidence: {np.max(prediction):.2f})")  # Debugging

                # Display the predicted gesture on screen
                cv2.putText(image, f'{predicted_label.upper()} ({np.max(prediction):.2f})',
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw landmarks
            draw_landmarks(image, results)

            # Show output
            cv2.imshow("Sign Language Detection", image)

            # Press 'q' to exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("[INFO] Exiting program...")
                break

except Exception as e:
    print(f"[ERROR] An exception occurred while running MediaPipe: {e}")

print("[INFO] Releasing resources...")
cap.release()
cv2.destroyAllWindows()
print("[INFO] Program finished successfully.")
