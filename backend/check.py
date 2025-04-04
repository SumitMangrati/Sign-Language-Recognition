import numpy as np
import tensorflow as tf

# Load your trained model (.h5 file)
model = tf.keras.models.load_model("action.h5")

# Create a dummy input with the expected sequence length (e.g., 30 frames)
dummy_input = np.zeros((1, 30, 1662))  # Try different feature sizes if needed

# Make a prediction
try:
    pred = model.predict(dummy_input)
    print("✅ Model successfully processed input!")
    print("Expected input shape:", model.input_shape)  # Check model's expected input
    print("Dummy input shape:", dummy_input.shape)  # Check the shape we are passing
except Exception as e:
    print("❌ Error:", e)
