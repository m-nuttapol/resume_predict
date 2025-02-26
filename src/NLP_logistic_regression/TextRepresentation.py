import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer

class Tfidf:
    def __init__(self, x_train, x_test, script_dir ,max_features=1000 , vectorizer_name="vectorizer.pkl"):
        """
        Initialize the TF-IDF vectorizer and store train/test data.
        """
        self.tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')

        self.train = x_train
        self.test = x_test

        self.model_path = os.path.abspath(os.path.join(script_dir, "../../src/model", vectorizer_name))


    def process(self):
        """Apply TF-IDF transformation correctly using both text and extracted skills."""

        # ✅ Ensure required columns exist
        if "cleaned_text" not in self.train.columns or "extract_skills" not in self.train.columns:
            raise ValueError("❌ 'cleaned_text' or 'extract_skills' column is missing!")

        # ✅ Merge `cleaned_text` and `extract_skills` into one text column
        self.train["merged_text"] = self.train["cleaned_text"] + " " + self.train["extract_skills"].apply(lambda skills: " ".join(skills))
        self.test["merged_text"] = self.test["cleaned_text"] + " " + self.test["extract_skills"].apply(lambda skills: " ".join(skills))

        # ✅ Apply TF-IDF on merged text
        X_train_tfidf = self.tfidf.fit_transform(self.train["merged_text"])
        X_test_tfidf = self.tfidf.transform(self.test["merged_text"])

        # ✅ Save the vectorizer
        joblib.dump(self.tfidf, self.model_path)
        print(f"✅ TF-IDF vectorizer saved at {self.model_path}")

        print("✅ After TF-IDF:")
        print("X_train_tfidf shape:", X_train_tfidf.shape)
        print("X_test_tfidf shape:", X_test_tfidf.shape)

        return X_train_tfidf, X_test_tfidf
