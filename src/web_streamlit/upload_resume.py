import streamlit as st
import pdfplumber
import docx2txt
import joblib
import os
import gspread
from dotenv import load_dotenv
from google.oauth2 import service_account
import json

# Load environment variables from .env file (ensure .env is present in your project)
# load_dotenv()

# Get credentials from environment variable (the path to the credentials file)
# credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# credentials_path = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# if credentials_path is None:
#     raise ValueError("Google credentials environment variable is not set")

# # Load credentials from the service account JSON
# credentials = service_account.Credentials.from_service_account_file(credentials_path)


# Load JSON credentials from Streamlit secrets
credentials_json = st.secrets["GOOGLE_CREDENTIALS"]
credentials_dict = json.loads(credentials_json)

# Authenticate with Google using service account info
credentials = service_account.Credentials.from_service_account_info(credentials_dict)

# Load the trained model, vectorizer, and label encoder
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "resume_model.pkl"))
vectorizer_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "vectorizer.pkl"))
label_encoder_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "label_encoder.pkl"))

# Ensure the model, vectorizer, and label encoder files exist before loading
if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path):
    model = joblib.load(model_path)  # Load the trained model
    vectorizer = joblib.load(vectorizer_path)  # Load the saved vectorizer
    label_encoder = joblib.load(label_encoder_path)  # Load the label encoder
    print("✅ Model, vectorizer, and label encoder loaded successfully.")
else:
    raise FileNotFoundError(f"❌ Model, vectorizer, or label encoder file not found.")

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

# Function to send data to Google Sheets
def send_to_google_sheet(extracted_text, predicted_role):
    # Authorize using Google credentials
    client = gspread.authorize(credentials)
    
    # Open the Google Sheet (replace with your sheet name or URL)
    sheet = client.open("DataSetStore").sheet1
    # Insert the extracted text and predicted role into the sheet
    sheet.append_row([predicted_role, extracted_text])
    st.success("✅ Data successfully sent to Google Sheet!")

# Streamlit UI
st.title("Resume Uploader and Job Role Predictor")

# Refresh button logic to clear outputs
if st.button("Clear Outputs"):
    # Reset the session state to clear uploaded file, text, and other variables
    st.session_state.clear()
    st.experimental_rerun()

uploaded_file = st.file_uploader("Upload a Resume (PDF or DOCX)", type=["pdf", "docx"])

# Process file if uploaded
if uploaded_file is not None:
    # Store the uploaded file in session state so it persists across reruns
    st.session_state.uploaded_file = uploaded_file

    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        extracted_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF or DOCX file.")
        extracted_text = ""

    # Store extracted text in session state
    st.session_state.extracted_text = extracted_text

    if extracted_text:
        st.subheader("Extracted Text:")
        st.text_area("", extracted_text, height=300)

        # Convert text to numerical features using the TF-IDF vectorizer
        extracted_text_vectorized = vectorizer.transform([extracted_text])

        # Make Prediction
        predicted_role_encoded = model.predict(extracted_text_vectorized)[0]

        # Convert encoded prediction back to job title
        predicted_role = label_encoder.inverse_transform([predicted_role_encoded])[0]

        # Store the predicted role in session state
        st.session_state.predicted_role = predicted_role

        # Display Prediction
        st.subheader("Predicted Job Role:")
        st.write(f"📌 **{predicted_role}**")

        # Ask for user consent
        consent = st.radio("Do you want to submit the predicted role to the Google Sheet?", ["Yes", "No"])

        if consent == "Yes":
            # Submit the predicted role and extracted text to Google Sheets
            if st.button("Submit to Google Sheet"):
                send_to_google_sheet(extracted_text, predicted_role)
                # Clear output and reset the state after submission
                st.session_state.clear()
                st.experimental_rerun()

        elif consent == "No":
            # Let the user manually input a role if they disagree with the prediction
            manual_role = st.text_input("Please specify the role you think is most relevant:")
            if manual_role:
                if st.button("Submit Manual Role to Google Sheet"):
                    send_to_google_sheet(extracted_text, manual_role)
                    st.success("✅ Manual role successfully submitted to Google Sheet!")
                    # Clear output and reset the state after submission
                    st.session_state.clear()
                    st.experimental_rerun()
