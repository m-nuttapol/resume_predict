import streamlit as st
import pdfplumber
import docx2txt
import joblib  # Use joblib instead of pickle
import os

# Load the trained model and vectorizer
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "resume_model.pkl"))
vectorizer_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "vectorizer.pkl"))

print("Model path:", model_path)
print("Vectorizer path:", vectorizer_path)

# Ensure the files exist before loading
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)  # Load the trained model
    vectorizer = joblib.load(vectorizer_path)  # Load the saved vectorizer
    print("✅ Model and vectorizer loaded successfully.")
else:
    raise FileNotFoundError(f"❌ Model or vectorizer file not found at {model_path} or {vectorizer_path}")

# Function to extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to extract text from a DOCX
def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)

# Streamlit UI
st.title("Resume Uploader and Job Role Predictor")

uploaded_file = st.file_uploader("Upload a Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        extracted_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF or DOCX file.")
        extracted_text = ""

    if extracted_text:
        st.subheader("Extracted Text:")
        st.text_area("", extracted_text, height=300)

        # **Convert text to numerical features using the TF-IDF vectorizer**
        extracted_text_vectorized = vectorizer.transform([extracted_text])

        # **Make Prediction**
        predicted_role = model.predict(extracted_text_vectorized)[0]
        
        # **Display Prediction**
        st.subheader("Predicted Job Role:")
        st.write(f"📌 **{predicted_role}**")
