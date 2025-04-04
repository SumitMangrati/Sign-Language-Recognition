from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model = tf.keras.models.load_model("action.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Actions list (update according to your model)
actions = ["hello", "thanks", "iloveyou"]  # Modify based on your dataset
sequence = []
predictions = []
threshold = 0.5

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to extract keypoints
def extract_keypoints(results):
    # Extract pose keypoints (132 values)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    # Extract left-hand keypoints (63 values)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # Extract right-hand keypoints (63 values)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    # Extract face keypoints (1404 values)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)

    # Combine all keypoints
    return np.concatenate([pose, lh, rh, face])  # Ensures total features = 1662


@app.route("/predict", methods=["POST"])
def predict():
    global sequence  # Add this line to access the global variable
    try:
        file = request.files["image"]
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Convert image to RGB for Mediapipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep only last 30 frames

            # Prediction logic
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                # Apply voting logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        predicted_action = actions[np.argmax(res)]
                    else:
                        predicted_action = "Unknown"
                else:
                    predicted_action = "Unknown"
            else:
                predicted_action = "Collecting frames..."

            return jsonify({"prediction": predicted_action})

    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
