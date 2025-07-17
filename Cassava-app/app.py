import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# ------------------- STREAMLIT SETTINGS -------------------
st.set_page_config(page_title="Cassava Mosaic Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>🍃 Cassava Mosaic Disease Detection System</h1>", unsafe_allow_html=True)
st.write("Upload a cassava leaf image to detect Cassava Mosaic Disease.")

# ------------------- CLASS NAMES -------------------
class_names = ['healthy', 'mosaic disease']

# ------------------- LOAD MODEL USING TFSMLayer -------------------
@st.cache_resource
def load_model():
    return tf.keras.layers.TFSMLayer('model.savedmodel', call_endpoint='serving_default')

with st.spinner('🔄 Loading disease detection model...'):
    layer = load_model()

# ------------------- PREDICTION FUNCTION -------------------
def import_and_predict(image_data, layer):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)
    image = np.asarray(image) / 255.0
    img_reshape = image[np.newaxis, ...]
    input_tensor = tf.convert_to_tensor(img_reshape)
    output = layer(input_tensor)
    predictions = output['dense'] if 'dense' in output else list(output.values())[0]
    return predictions.numpy()

# ------------------- UI LAYOUT -------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    uploaded_file = st.file_uploader("📤 Upload a cassava leaf image", type=["jpg", "jpeg", "png"])

with right_col:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Cassava Leaf", width=200)


        prediction = import_and_predict(image, layer)
        score = tf.nn.softmax(prediction[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.markdown(f"###  Prediction: **{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        if predicted_class == "mosaic disease":
            st.error("⚠️ Cassava Mosaic Disease Detected!")
            st.markdown("###  Recommended Actions:")
            st.markdown("""
            -  **Inspect fields regularly** for early signs of infection.
            -  **Remove and destroy infected plants** immediately.
            -  **Use disease-resistant cassava varieties.**
            -  **Avoid replanting with cuttings from infected plants.**
            -  **Control whitefly vectors** through integrated pest management.
            -  **Sanitize farming tools** to prevent disease spread.
            """)
        else:
            st.success("✅ The cassava leaf appears healthy.")
            st.markdown("###  Best Practices for Cassava Health:")
            st.markdown("""
            -  **Plant certified, disease-free cuttings.**
            -  **Ensure optimal growing conditions and soil health.**
            -  **Monitor and manage pest populations.**
            -  **Train field workers to recognize disease symptoms early.**
            -  **Rotate crops** to reduce the risk of disease buildup.
            """)
    else:
        st.info("📎 Please upload a cassava leaf image from the left panel to begin.")
