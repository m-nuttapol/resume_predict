'''Step 1 : Import essential libraries'''
import pandas as pd # For data processing and reading CSV files
import re # For regular expressions operations
import nltk
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # For splitting the dataset
from sklearn.feature_extraction.text import TfidfVectorizer # For text vectorization
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import sys
from sklearn.metrics import classification_report
import numpy as np

from nltk.util import ngrams
from fuzzywuzzy import fuzz
from fuzzywuzzy import process



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from utils import save_df_to_csv, load_skills_from_json




# ✅ Initialize and run processing in one step
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../../src/dataset/all_resume_dataset.csv")

'''Step 2 : Load the dataset'''
# reading CSV files
# file_path = 'all_resume_dataset.csv'
df = pd.read_csv(file_path)

print(df.head())
print(df.shape)

# '''Step 3 : Preprocess data'''
# Drop not using columns
# df.drop(columns=['ID','Resume_html'], inplace=True)

# # Check null values
print('The number of null values for each column : ',df.isnull().sum())
# Drop rows where any columns has a null value
df = df.dropna()


# # Check duplicated data
print('The number of duplicated rows : ',df.duplicated().sum())
df = df.drop_duplicates(keep='first')

# # Check data
print(df.info())

# # Clean the Resume column using Regex approach
def expand_contractions(text, contractions_dict = {
    "I'm": "I am",
    "can't": "cannot",
    "won't": "will not",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "they're": "they are",
    "you're": "you are",
    "we're": "we are",
    "isn't": "is not",
    "aren't": "are not",
    "doesn't": "does not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "mustn't": "must not",
    "let's": "let us"
}):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions_dict.keys()) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group()], text)

# # Removing Emails, Phone Numbers, URLs
def remove_personal_info(text):
    text = re.sub(r'\S+@\S+', ' ', text) # Remove emails
    text = re.sub(r'\b\d{10,15}\b', ' ', text) # Remove phone numbers
    text = re.sub(r'http\S+|www\S+', ' ', text)  # Remove URLs (LinkedIn or other websites)
    return text

def clean_text(text):
    text = text.lower() # Converts all characters in the string to lowercase
    text = re.sub(r'[^a-z\s]',' ',text) # removes punctuation, numbers, and special characters
    text = re.sub(r'\s+', ' ', text).strip() # Replaces extra space
    return text

# # Apply basic preprocessing
df['cleaned_text'] = df['Extracted Text'].apply(expand_contractions)
df['cleaned_text'] = df['Extracted Text'].apply(remove_personal_info)
df['cleaned_text'] = df['Extracted Text'].apply(clean_text)

# # Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# # Initialize stopwords and Lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# # Text cleaning function
def nltk_preprocess(text):
    tokens = word_tokenize(text) # Tokenization
    tokens = [word for word in tokens if word not in stop_words] # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # Lemmatization
    return " ".join(tokens)

# # Apply NLTK preprocessing
df['cleaned_text'] = df['cleaned_text'].apply(nltk_preprocess)

## load skill
config_file_path = os.path.join(script_dir, "../../src/config.json")
skills_dict, skills_list = load_skills_from_json(config_file_path)

## extract skill


def extract_skills_fast(text, skills_list, threshold=80):
    """
    Extracts skills from a given text using substring search and fuzzy matching.

    Parameters:
    - text (str): Cleaned job description or resume.
    - skills_list (list): A predefined list of skills to match.
    - threshold (int): Minimum similarity score for fuzzy matching.

    Returns:
    - list: Extracted skills found in the text.
    """
    if not isinstance(text, str):
        return []

    # Convert everything to lowercase
    text = text.lower()
    print(text)
    skills_list = [skill.lower() for skill in skills_list]

    extracted_skills = set()

    # Fast substring matching for exact phrases
    for skill in skills_list:
        if skill in text:  # Direct match
            extracted_skills.add(skill)
        else:
            # Use fuzzy matching only if an exact match is not found
            match, score = process.extractOne(skill, text.split())  # Compare word by word
            if score >= threshold:
                extracted_skills.add(skill)
                print("1111",skill)

    return list(extracted_skills)

df['extract_skills'] = df['cleaned_text'].apply(lambda x: extract_skills_fast(x, skills_list))


# # Save cleaned data to CSV file
df.to_csv('Cleaned_Resume.csv', index=False)
save_df_to_csv(df, "Cleaned_Resume.csv")

# # Check class balance
job_role_counts = df['Job Role'].value_counts()

plt.figure(figsize=(10, 5))
sns.barplot(x=job_role_counts.index, y=job_role_counts.values)
plt.xticks(rotation=90)
plt.xlabel("Job Category")
plt.ylabel("Count")
plt.title("Job Category Distribution")
plt.show()

proportion_largest_smallest = job_role_counts[0]/job_role_counts[-1]
total_samples = len(df)
category_percentages = (job_role_counts / total_samples) * 100
print('job_role_counts : ',job_role_counts)
print('ratio_largest_smallest : ',proportion_largest_smallest)

# '''Step 5 : Encode job categories'''
le = LabelEncoder()
df['job_role_encoded'] = le.fit_transform(df['Job Role'])
# print(dict(zip(le.classes_,le.transform(le.classes_))))

# '''Step 6 : Split data into train and test sets'''
X = df['cleaned_text']
y = df['job_role_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# '''Step 7 : Convert Text to Features (TF-IDF)'''
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train) # Fit and transform training data
X_test_tfidf = tfidf.transform(X_test)

# '''Step 8 : Train Model'''
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)  
print(classification_report(y_test, y_pred))
