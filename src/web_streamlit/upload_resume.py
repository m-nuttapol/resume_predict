import streamlit as st
import pdfplumber
import docx2txt
import joblib
import os
import gspread
from google.oauth2 import service_account

# Load Google credentials from Streamlit secrets
google_credentials = st.secrets["GOOGLE_CREDENTIALS"]

google_credentials_dict = {
    "type": google_credentials["type"],
    "project_id": google_credentials["project_id"],
    "private_key_id": google_credentials["private_key_id"],
    "private_key": google_credentials["private_key"],  
    "client_email": google_credentials["client_email"],
    "client_id": google_credentials["client_id"],
    "auth_uri": google_credentials["auth_uri"],
    "token_uri": google_credentials["token_uri"],
    "auth_provider_x509_cert_url": google_credentials["auth_provider_x509_cert_url"],
    "client_x509_cert_url": google_credentials["client_x509_cert_url"],
    "universe_domain": google_credentials["universe_domain"]
}

# Define the required scopes
scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]

# Authenticate with Google using service account info and correct scopes
credentials = service_account.Credentials.from_service_account_info(
    google_credentials_dict, scopes=scopes
)

# Verify the authentication
st.write("Successfully authenticated with Google!")

# Load the trained model, vectorizer, and label encoder
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "resume_model.pkl"))
vectorizer_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "vectorizer.pkl"))
label_encoder_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "label_encoder.pkl"))

if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path):
    model = joblib.load(model_path)  # Load the trained model
    vectorizer = joblib.load(vectorizer_path)  # Load the saved vectorizer
    label_encoder = joblib.load(label_encoder_path)  # Load the label encoder
    st.write("✅ Model, vectorizer, and label encoder loaded successfully.")
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
    client = gspread.authorize(credentials)
    
    # Debugging logs
    st.write("Attempting to send data to Google Sheets...")
    
    try:
        st.write("Opening the sheet...")

        # Open the sheet
        sheet = client.open("DataSetStore").sheet1
        
        # Attempt to append the data
        st.write("Appending data to the sheet...")
        response = sheet.append_row([predicted_role, extracted_text])
        st.write(f"API Response: {response}")
        
        st.success("✅ Data successfully sent to Google Sheet!")
    except Exception as e:
        st.error(f"❌ Failed to send data to Google Sheets: {str(e)}")

# Streamlit UI
st.title("Resume Uploader and Job Role Predictor")

# Refresh button logic to clear outputs
if st.button("Clear Outputs"):
    st.session_state.clear()

uploaded_file = st.file_uploader("Upload a Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        extracted_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF or DOCX file.")
        extracted_text = ""

    st.session_state.extracted_text = extracted_text

    if extracted_text:
        st.subheader("Extracted Text:")
        st.text_area("", extracted_text, height=300)

        extracted_text_vectorized = vectorizer.transform([extracted_text])

        predicted_role_encoded = model.predict(extracted_text_vectorized)[0]

        predicted_role = label_encoder.inverse_transform([predicted_role_encoded])[0]

        st.session_state.predicted_role = predicted_role

        st.subheader("Predicted Job Role:")
        st.write(f"📌 **{predicted_role}**")

        consent = st.radio("Do you want to submit the predicted role to the Google Sheet?", ["Yes", "No"])

        if consent == "Yes":
            if st.button("Submit to Google Sheet"):
                send_to_google_sheet(extracted_text, predicted_role)

        elif consent == "No":
            manual_role = st.text_input("Please specify the role you think is most relevant:")
            if manual_role:
                if st.button("Submit Manual Role to Google Sheet"):
                    send_to_google_sheet(extracted_text, manual_role)
