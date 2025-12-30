# import numpy as np
# import streamlit as st
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import cv2
# from PIL import Image

# # -------------------------------------------------------------
# # Page config
# # -------------------------------------------------------------
# st.set_page_config(page_title="Alzheimer MRI Detection", layout="centered")

# st.title("🧠 Alzheimer’s Disease Detection (MRI)")
# st.write("Upload an MRI image to predict the Alzheimer’s stage.")

# # -------------------------------------------------------------
# # Load trained model
# # -------------------------------------------------------------
# @st.cache_resource
# def load_trained_model():
#     return load_model(
#         r"c:\Users\saike\Downloads\medimg_py397\Final_final_reduced.keras",
#         compile=False
#     )

# model = load_trained_model()

# # -------------------------------------------------------------
# # Class labels (same order as training)
# # -------------------------------------------------------------
# class_labels = [
#     "Mild Impairment",
#     "Moderate Impairment",
#     "No Impairment",
#     "Very Mild Impairment"
# ]

# # -------------------------------------------------------------
# # Image preprocessing (same as training)
# # -------------------------------------------------------------
# def load_and_prepare_image_from_upload(uploaded_file, target_size=(128, 128)):
#     image_pil = Image.open(uploaded_file).convert("L")  # grayscale
#     img = np.array(image_pil)

#     img = cv2.resize(img, target_size)
#     img = img.astype("float32") / 255.0

#     img = np.expand_dims(img, axis=-1)   # (128,128,1)
#     img = np.expand_dims(img, axis=0)    # (1,128,128,1)

#     return img, image_pil

# # -------------------------------------------------------------
# # Upload image
# # -------------------------------------------------------------
# uploaded_file = st.file_uploader(
#     "Upload MRI Image",
#     type=["jpg", "jpeg", "png"]
# )

# # -------------------------------------------------------------
# # Prediction
# # -------------------------------------------------------------
# if uploaded_file is not None:
#     st.subheader("Uploaded Image")
#     st.image(uploaded_file, width=300)

#     img_array, original_image = load_and_prepare_image_from_upload(uploaded_file)

#     prediction = model.predict(img_array)[0]

#     st.subheader("Prediction Probabilities")
#     for i, cls in enumerate(class_labels):
#         st.write(f"{cls}: **{prediction[i] * 100:.2f}%**")

#     predicted_class = np.argmax(prediction)
#     confidence = prediction[predicted_class] * 100

#     st.subheader("Final Prediction")
#     st.success(
#         f"Predicted Stage: **{class_labels[predicted_class]}**\n\n"
#         f"Confidence: **{confidence:.2f}%**"
#     )

#     # Show grayscale image with title
#     fig, ax = plt.subplots()
#     ax.imshow(original_image, cmap="gray")
#     ax.set_title(f"{class_labels[predicted_class]} ({confidence:.2f}%)")
#     ax.axis("off")
#     st.pyplot(fig)










import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# =============================================================
# Page Configuration
# =============================================================
st.set_page_config(
    page_title="Alzheimer MRI Detection",
    page_icon="🧠",
    layout="centered"
)

# =============================================================
# Custom CSS for Colorful UI
# =============================================================
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.main {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
}
h1 {
    color: #2C3E50;
    text-align: center;
}
h2, h3 {
    color: #34495E;
}
.result-box {
    background-color: #1E88E5;   /* Blue background */
    padding: 18px;
    border-radius: 10px;
    border-left: 6px solid #0D47A1;
    margin-top: 15px;
    color: #FFFFFF;              /* White text */
}
.result-box h3 {
    color: #FFFFFF;
    margin-bottom: 10px;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================
# Sidebar
# =============================================================
st.sidebar.title("🧠 Alzheimer MRI App")
st.sidebar.write("Final Year Project")
st.sidebar.markdown("---")
st.sidebar.write("**Steps to Use:**")
st.sidebar.write("1. Upload MRI Image")
st.sidebar.write("2. Model analyzes image")
st.sidebar.write("3. Disease stage is predicted")
st.sidebar.markdown("---")
st.sidebar.write("CNN-based Alzheimer Detection")

# =============================================================
# Main Title
# =============================================================
st.title("Alzheimer’s Disease Detection Using MRI")
st.write(
    "This application uses a **Convolutional Neural Network (CNN)** "
    "to detect Alzheimer’s disease stages from MRI brain images."
)

# =============================================================
# Load Model
# =============================================================
@st.cache_resource
def load_trained_model():
    return load_model(
        r"c:\Users\saike\Downloads\medimg_py397\Final_final_reduced.keras",
        compile=False
    )

model = load_trained_model()

# =============================================================
# Class Labels (Same order as training)
# =============================================================
class_labels = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment"
]

# =============================================================
# Image Preprocessing (Same as Training)
# =============================================================
def preprocess_uploaded_image(uploaded_file, target_size=(128, 128)):
    image_pil = Image.open(uploaded_file).convert("L")  # grayscale
    img = np.array(image_pil)

    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=-1)   # (128,128,1)
    img = np.expand_dims(img, axis=0)    # (1,128,128,1)

    return img, image_pil

# =============================================================
# Image Upload
# =============================================================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# =============================================================
# Prediction
# =============================================================
if uploaded_file is not None:
    st.subheader("Uploaded MRI Image")
    st.image(uploaded_file, width=300)

    img_array, original_image = preprocess_uploaded_image(uploaded_file)

    prediction = model.predict(img_array)[0]

    st.subheader("Prediction Confidence (All Classes)")
    for i, cls in enumerate(class_labels):
        st.write(cls)
        st.progress(float(prediction[i]))

    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100

    # =============================================================
    # Final Result Box
    # =============================================================
    st.markdown(
        f"""
        <div class="result-box">
            <h3>Final Prediction</h3>
            <b>Predicted Stage:</b> {class_labels[predicted_class]} <br>
            <b>Confidence:</b> {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    # =============================================================
    # Display Image with Prediction
    # =============================================================
    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap="gray")
    ax.set_title(f"{class_labels[predicted_class]} ({confidence:.2f}%)")
    ax.axis("off")
    st.pyplot(fig)

# =============================================================
# Footer
# =============================================================
st.markdown(
    "<div class='footer'>Developed for Academic Demonstration | Alzheimer MRI Classification</div>",
    unsafe_allow_html=True
)
