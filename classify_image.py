import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained MobileNet model
model = MobileNet(weights='imagenet')

# Load and preprocess image
img_path = 'your_image.jpg'  # Replace with your image filename
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict
preds = model.predict(x)
results = decode_predictions(preds, top=3)[0]

# Display predictions
print("Top Predictions:")
for i, (imagenet_id, label, prob) in enumerate(results):
    print(f"{i + 1}. {label}: {prob * 100:.2f}%")
