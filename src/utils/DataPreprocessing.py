import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Fix: Download only necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataCleaning:
    def __init__(self, df):
        """Initialize ResumeProcessor with dataset path"""
        self.df = df
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def remove_personal_info(self, text):
        """Removes emails, phone numbers, and URLs from text before cleaning"""
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'\b\d{10,15}\b', '', text)  # Remove phone numbers
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        
        return text
    
    def expand_contractions(self,text, contractions_dict = {
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


    # # Text cleaning function
    def nltk_preprocess(self,text):
        tokens = word_tokenize(text) # Tokenization
        tokens = [word for word in tokens if word not in self.stop_words] # Remove stopwords
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens] # Lemmatization
        return " ".join(tokens)

    def clean_text(self, text):
        """Cleans text: removes special characters, stopwords, and lemmatizes words"""
        if not isinstance(text, str):
            return ""

        text = self.remove_personal_info(text)  
        text = text.lower()  # Convert to lowercase
        text = self.expand_contractions(text)  
        text = re.sub(r'&[a-z]+;', ' ', text)  # Remove HTML entities (&amp;)
        text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters also number
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

        return text
    

    def nltk_preprocess(self,text):
        tokens = word_tokenize(text) # Tokenization
        tokens = [word for word in tokens if word not in self.stop_words] # Remove stopwords
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens] # Lemmatization
        return " ".join(tokens)




    def preprocess_resumes(self):
        """Applies cleaning functions to Resume column"""
        if self.df is not None and "Resume" in self.df.columns:

            self.df["cleaned_text"] = self.df["Resume"].apply(self.clean_text)
            self.df['cleaned_text'] = self.df['cleaned_text'].apply(self.nltk_preprocess)

            print("✅ Resumes cleaned successfully.")
        else:
            print("❌ Error: 'Resume' column not found in dataset.")



    def process(self):
        """Runs full processing and returns the cleaned DataFrame"""

        if self.df is not None:
            print('The number of null values for each column : ',self.df.isnull().sum())
            # Drop rows where any columns has a null value
            self.df = self.df.dropna()


            # # Check duplicated data
            print('The number of duplicated rows : ',self.df.duplicated().sum())
            self.df = self.df.drop_duplicates(keep='first')

            print('Total Row : ',len(self.df))

            self.preprocess_resumes()
            # save_df_to_csv(self.df, "cleaned_dataset.csv")
            # save_df_to_excel(self.df, "cleaned_dataset.xlsx")
        return self.df  # ✅ Fix: Return cleaned DataFrame
