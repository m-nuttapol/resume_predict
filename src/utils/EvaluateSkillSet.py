import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import save_df_to_csv, save_df_to_excel, load_role_and_skills ,apply_roles
import nltk
import os
from fuzzywuzzy import fuzz, process

# Manually set the path to your NLTK data
nltk_data_path = os.path.expanduser("~/nltk_data")
nltk.data.path.append(nltk_data_path)

# Reinstall the correct models
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)


class EvaluateSkill:
    def __init__(self, df, config_path):
        """Initialize with DataFrame."""
        self.df = df
        self.position_role ,self.skills_list = load_role_and_skills(config_path)
        self.skills_list = [skill.lower() for skill in self.skills_list]
        print(self.skills_list)


    def extract_skills_fast(self,text, threshold=90):
        """
        Extracts relevant skills from a given text using exact match and improved fuzzy matching.

        Parameters:
        - text (str): Cleaned job description or resume.
        - skills_list (list): Predefined list of skills.
        - threshold (int): Minimum similarity score for fuzzy matching.

        Returns:
        - list: Extracted skills found in the text.
        """
        if not isinstance(text, str):
            print("111111")
            print(text)
            return []

        # Convert text to lowercase and tokenize words
        text = text.lower()
        print(text)
        extracted_skills = set()

        # Exact phrase matching for high accuracy
        for skill in self.skills_list:
            skill_lower = skill.lower()
            
            if f" {skill_lower} " in f" {text} ":  # Ensures full-word match
                extracted_skills.add(skill)
                print("1111",skill)
            
            else:
                # Apply fuzzy matching only for multi-word skills
                if " " in skill_lower:
                    match, score = process.extractOne(skill_lower, text.split(), scorer=fuzz.token_sort_ratio)
                    if score >= threshold:
                        extracted_skills.add(skill)
        
        return list(extracted_skills)

    def process(self):
        """Applies cleaning, extracts skills dynamically, and prepares data."""
        required_columns = {"Category", "cleaned_text"}

        if self.df is None or not required_columns.issubset(self.df.columns):
                raise ValueError("❌ Data not loaded or required columns are missing!")

        self.df = apply_roles(self.df, self.position_role)
        self.df['extract_skills'] = self.df['cleaned_text'].apply(lambda x: self.extract_skills_fast(x))

        save_df_to_csv(self.df, "cleaned_dataset.csv")
        save_df_to_excel(self.df, "cleaned_dataset.xlsx")

        print("✅ Data Evaluation Completed.")
        return self.df  # ✅ Return processed DataFrame
