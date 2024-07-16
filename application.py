import streamlit as st
from src.evaluate import evaluate_model
from src.predict import predict_image
import os

st.title("Model Evaluation and Prediction")

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Select a page:", ["Predict", "Evaluate"])

if option == "Predict":
    st.header("Predict Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        temp_dir = "upload_folder"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write(f"File uploaded successfully: {uploaded_file.name}")

        result = predict_image(file_path)

        st.write("Prediction result:")
        st.json(result)

        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

elif option == "Evaluate":
    st.header("Evaluate Model")
    if st.button("Evaluate"):
        result = evaluate_model()
        st.write(f"{result}")
