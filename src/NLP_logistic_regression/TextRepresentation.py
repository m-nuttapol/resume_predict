from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

class Tfidf:
    def __init__(self, x_train, x_test, max_features=1000):
        """
        Initialize the TF-IDF vectorizer and store train/test data.
        """
        self.tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')

        self.train = x_train
        self.test = x_test

    def apply_smote(X_train_tfidf, y_train):
        """Apply SMOTE to handle class imbalance."""
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

        print("✅ After SMOTE:")
        print("X_resampled shape:", X_resampled.shape)
        print("y_resampled shape:", y_resampled.shape)

        return X_resampled, y_resampled

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


        print("✅ After TF-IDF:")
        print("X_train_tfidf shape:", X_train_tfidf.shape)
        print("X_test_tfidf shape:", X_test_tfidf.shape)

        return X_train_tfidf, X_test_tfidf

