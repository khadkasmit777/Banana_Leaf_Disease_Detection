# Import necessary libraries
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load the trained model
model = load_model('Banana_Disease_Detection_Model.h5')

# Define the class labels (based on your dataset order)
class_names = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']

# Streamlit app title and description
st.title("🍌 Banana Leaf Disease Detection")
st.markdown("Upload a banana leaf image. The model will detect its condition only if it's a valid leaf.")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    try:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        if opencv_image is None:
            st.error("❌ Could not read image. Please upload a valid image.")
        else:
            # Resize for smaller display
            display_image = cv2.resize(opencv_image, (250, 250))
            st.image(display_image, channels="BGR", caption="Uploaded Image", use_column_width=False)

            # Detect green area to verify if it's likely a leaf
            hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = np.sum(mask > 0) / mask.size

            if green_ratio < 0.10:  
                st.error("❌ This does not appear to be a banana leaf image. Please upload a valid leaf photo.")
            else:
                # Show detect button only if green area is enough
                if st.button("Detect"):
                    # Preprocess image
                    resized_img = cv2.resize(opencv_image, (224, 224))
                    normalized_img = resized_img / 255.0
                    input_img = np.expand_dims(normalized_img, axis=0)

                    # Prediction
                    prediction = model.predict(input_img)
                    predicted_class = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100

                    # Display result
                    st.success(f"🌿 Detection: **{predicted_class.capitalize()}** leaf")
                    st.info(f"Confidence: **{confidence:.2f}%**")

                    # Class probabilities
                    st.subheader("Class Probabilities:")
                    for idx, cls in enumerate(class_names):
                        st.write(f"{cls.capitalize()}: {prediction[0][idx]*100:.2f}%")

    except Exception as e:
        st.error("❌ An error occurred while processing the image. Please upload a valid banana leaf image.")
