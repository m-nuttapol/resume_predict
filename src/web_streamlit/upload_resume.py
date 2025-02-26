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
# st.write("Successfully authenticated with Google!")

# Load the trained model, vectorizer, and label encoder
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "resume_model.pkl"))
vectorizer_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "vectorizer.pkl"))
label_encoder_path = os.path.abspath(os.path.join(script_dir, "../../src/model", "label_encoder.pkl"))

if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path):
    model = joblib.load(model_path)  # Load the trained model
    vectorizer = joblib.load(vectorizer_path)  # Load the saved vectorizer
    label_encoder = joblib.load(label_encoder_path)  # Load the label encoder
    # st.write("✅ Model, vectorizer, and label encoder loaded successfully.")
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
    
    try:
        # Check if the sheet exists, create if not
        try: 
            sheet = client.open("Resume DataSet").sheet1
        except gspread.SpreadsheetNotFound:
            sheet = client.create("Resume DataSet").sheet1
        
        row = [predicted_role, extracted_text]
        
        # Get the last row number and add data in the next row
        last_row = len(sheet.get_all_values()) + 1
        sheet.update(f"A{last_row}:B{last_row}", [row])
        
        st.success("✅ Data successfully sent to Google Sheet!")
    except Exception as e:
        st.error(f"❌ Failed to send data to Google Sheets: {str(e)}")

def page1():
# Streamlit UI
    st.title("Resume Uploader and Job Role Predictor")


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


            # Radio button to ask user if they want to submit the data
            consent = st.radio(
                "Would you like to help us support enhancing the prediction by providing this data?",
                ["Yes, I would like to help!", "Yes, but I would prefer to provide my own answer", "No, thank you"]
            )

            # Initialize a flag for submission
            submit_flag = False

            if consent == "Yes, I would like to help!":
                submit_flag = st.button("Submit")

            elif consent == "Yes, but I would prefer to provide my own answer":
                manual_role = st.text_input("Please specify the role you think is most relevant:")
                if manual_role:
                    submit_flag = st.button("Submit")

            elif consent == "No, thank you":
                submit_flag = st.button("Submit")

            # Once user has chosen and clicked submit, proceed with the form submission
            if submit_flag:
                if consent == "Yes, I would like to help!":
                    # Simulate submitting data
                    send_to_google_sheet(extracted_text, predicted_role)
                    st.success("✅ You chose to submit the predicted role. Thank you!")
                    st.session_state.current_page = "thank_you_page"

                elif consent == "Yes, but I would prefer to provide my own answer":
                    # Simulate submitting data
                    send_to_google_sheet(extracted_text, manual_role)
                    st.success("✅ Your manual role has been submitted. Thank you!")
                    st.session_state.current_page = "thank_you_page"
                elif consent == "No, thank you":
                    st.success("✅ You chose not to submit the data.")
                    st.session_state.current_page = "thank_you_page"

                st.experimental_rerun()


def thank_you_page():
    st.title("Thank You :D")
    st.write("Thank you for your input! Your response has been received.")

# Main app logic for page navigation
def main():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "page_1"  # Default to Page 1

    if st.session_state.current_page == "page_1":
        page1()  # Show the Resume Uploader page

    elif st.session_state.current_page == "thank_you_page":
        thank_you_page()  # Show the Thank You page

# Run the main function
if __name__ == "__main__":
    main()