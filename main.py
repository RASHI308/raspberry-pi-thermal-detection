import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from mlx90640 import MLX90640

# Load trained model
model = tf.keras.models.load_model("your_model_path.h5")

# Initialize MLX90640 thermal camera
mlx = MLX90640()
mlx.setup()

# Store last 5 frames for temperature smoothing
temp_history = deque(maxlen=5)

def get_thermal_data():
    """Capture a thermal frame and extract smoothed temperature values"""
    frame = np.zeros((24 * 32,))
    try:
        mlx.getFrame(frame)
        frame = np.reshape(frame, (24, 32))  # MLX90640 resolution is 24x32
        
        # Get temperature statistics
        max_temp = np.max(frame)
        min_temp = np.min(frame)
        avg_temp = np.mean(frame)
        
        temp_history.append([max_temp, min_temp, avg_temp])  # Store values
        
        # Compute moving average over last few frames
        smoothed_temp = np.mean(temp_history, axis=0)
        
        # Resize and preprocess frame for visualization
        vis_frame = cv2.resize(frame, (150, 150))
        vis_frame = cv2.normalize(vis_frame, None, 0, 255, cv2.NORM_MINMAX)
        vis_frame = vis_frame.astype(np.uint8)
        vis_frame = cv2.applyColorMap(vis_frame, cv2.COLORMAP_JET)
        
        return vis_frame, smoothed_temp  # Return smoothed temperature values
    
    except:
        return None, None

def preprocess_image(image):
    """Prepares image for model prediction"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (150, 150))  # Resize to match model input
    normalized = resized / 255.0
    return normalized.reshape(1, 150, 150, 1)

# Start detection
while True:
    thermal_image, temp_data = get_thermal_data()
    
    if thermal_image is None:
        print("Thermal camera error. Skipping frame.")
        continue

    input_img = preprocess_image(thermal_image)

    # Model prediction
    prediction = model.predict([input_img, temp_data])[0][0]

    # Dynamic confidence threshold
    threshold = 0.6 if np.mean(temp_data) > 30 else 0.5  

    if prediction > threshold:
        print(f"✅ Human Detected | Confidence: {prediction:.2f}")
    else:
        print(f"❌ No Human Detected | Confidence: {prediction:.2f}")

    # Show thermal image
    cv2.imshow("Thermal Feed", thermal_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()