import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import tempfile

st.title("Brain Tumor Detection")
st.write("Upload a brain scan image (JPG format) to detect tumor type.")

# File uploader
uploaded_file = st.file_uploader("Choose a JPG file", type=["jpg"])

# Load model once
@st.cache_resource
def load_cnn_model():
    model = load_model('model_best.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_cnn_model()

# Labels
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Filename:", uploaded_file.name)
    
    # Save the uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    
    # Load the image
    img = cv2.imread(temp_path)
    img = cv2.resize(img, (150, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    # Make prediction
    pred = model.predict(img)
    predicted_label = np.argmax(pred, axis=1)[0]
    predicted_class = labels[predicted_label]

    # Show result
    st.subheader("Detection")
    st.write(f"Detected Tumor Type: **{predicted_class}**")
