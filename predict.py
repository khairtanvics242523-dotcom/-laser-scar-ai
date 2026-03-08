import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("Loading model...")

# Load trained model
model = load_model("scar_model.h5")

# Path to test image
image_path = "test.jpg"

print("Loading image...")

# Read image
img = cv2.imread(image_path)

# Resize image
img = cv2.resize(img, (128,128))

# Normalize
img = img / 255.0

# Reshape for model
img = np.reshape(img, (1,128,128,3))

print("Running prediction...")

prediction = model.predict(img)

print("Prediction value:", prediction)

if prediction > 0.5:
    print("⚠️ Adverse Scar Reaction Detected")
else:
    print("✅ Safe Healing Scar")