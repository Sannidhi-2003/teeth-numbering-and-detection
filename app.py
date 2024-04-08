import os
import subprocess
import pandas as pd
import streamlit as st

def save_uploaded_file(file):
    os.makedirs("temp", exist_ok=True)

    file_path = os.path.join("temp", file.name)
    with open(file_path, "wb") as f:
        f.write(file.getvalue())

    return file_path


def run_detection(file_path,detect_script_path):

    command = [
        "python",
        detect_script_path,
        "--weights", "best.pt",
        "--img", str(256),
        "--conf", str(0.1),
        "--source", file_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    # st.text(result.stdout)
    st.text(result.stderr)

    result1=str(result)

    st.text("Accuracy")

    data = {
        'Class': ['all', 'abutment', 'canine', 'crown', 'implant', 'implant_minus', 'implant_plus', 'incisor', 'inlay', 'molar', 'premolar'],
        'Images': [511, 511, 511, 511, 511, 511, 511, 511, 511, 511, 511],
        'Instances': [5389, 68, 791, 60, 545, 74, 34, 1515, 1, 892, 1409],
        'Precision': [0.783, 1.0, 0.778, 1.0, 0.768, 0.0287, 1.0, 0.655, 1.0, 0.843, 0.756],
        'Recall': [0.374, 0.0, 0.777, 0.0, 0.369, 0.00155, 0.0, 0.78, 0.0, 0.934, 0.879],
        'mAP50': [0.405, 0.00977, 0.823, 0.0202, 0.494, 0.0663, 0.0337, 0.778, 0.00226, 0.949, 0.877],
        'mAP50-95': [0.169, 0.00341, 0.348, 0.00998, 0.164, 0.0219, 0.0124, 0.299, 0.000678, 0.464, 0.362]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Render the DataFrame as a table in Streamlit
    st.dataframe(df)



st.header('Teeth Detection')

file = st.file_uploader("Upload a file...", type=["jpg", "jpeg", "png"])

if file is not None:
    st.write("Uploaded File:")
    if file.type.startswith('image'):
        st.image(file, caption='Uploaded Image')
        folder = 'temp_images'
    else:
        st.error("Unsupported file format.")
        st.stop()

    if st.button("Detect!"):
        file_path = save_uploaded_file(file)
        detect_script_path = "yolov5\\detect.py"
        run_detection(file_path, detect_script_path)