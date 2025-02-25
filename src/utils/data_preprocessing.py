import re
from utils import load_csv
from utils import save_df_to_csv

class DataCleaning:
    def __init__(self, file_path=None):
        """Initialize ResumeProcessor with dataset path"""
        self.df = None
        self.file_path = file_path
        
    def clean_text(self, text):
        """Cleans text by removing special characters and lowercasing"""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_personal_info(self, text):
        """Removes emails, phone numbers, and URLs from text"""
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'\b\d{10,15}\b', '', text)  # Remove phone numbers
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        return text

    def preprocess_resumes(self):
        """Applies cleaning functions to Resume column"""
        if self.df is not None and "Resume" in self.df.columns:
            self.df["Cleaned_Resume"] = self.df["Resume"].apply(self.clean_text)
            self.df["Cleaned_Resume"] = self.df["Cleaned_Resume"].apply(self.remove_personal_info)
            print("✅ Resumes cleaned successfully.")
        else:
            print("❌ Error: 'Resume' column not found in dataset.")

    def process(self):
        """Runs full processing and returns the cleaned DataFrame"""
        self.df = load_csv(self.file_path)
        if self.df is not None:
            self.preprocess_resumes()
            save_df_to_csv(self.df)
