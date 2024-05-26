import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

import streamlit as st
import numpy as np
from keras.models import load_model
from keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps

# Custom DepthwiseConv2D class to handle 'groups' parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

# Register the custom object
custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}

# Define paths
model_file_path = os.path.join(os.path.dirname(__file__), "keras_model.h5")
labels_file_path = os.path.join(os.path.dirname(__file__), "labels.txt")

# Check if the files are present
if not os.path.exists(model_file_path):
    st.error("Failed to find the model file 'keras_model.h5'")
    st.stop()

if not os.path.exists(labels_file_path):
    st.error("Failed to find the labels file 'labels.txt'")
    st.stop()

# Load the model with custom objects
try:
    model = load_model(model_file_path, custom_objects=custom_objects, compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the class names
try:
    with open(labels_file_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    st.error(f"Error loading labels: {e}")
    st.stop()

# Define a function to predict the class
def predict(image):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data[0] = normalized_image_array
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name[2:], confidence_score, prediction[0]

# Streamlit UI
st.title("Osteoporosis Detection")

st.write("Upload an image to classify it as having Osteoporosis or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        class_name, confidence_score, all_predictions = predict(image)
        
        st.write(f"**Class**: {class_name}")
        st.write(f"**Confidence Score**: {confidence_score:.2f}")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
