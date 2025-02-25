from sklearn.feature_extraction.text import TfidfVectorizer

class Tfidf:
    def __init__(self, x_train,x_test):
        """
        Initialize the TfidfProcessor with max features and stop words.
        """
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

        self.train = x_train
        self.test = x_test

        #for trains
    def fit_transform(self, X_train):
        """
        Fit the TF-IDF vectorizer on the training data and transform it.
        """
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.is_fitted = True
        return X_train_tfidf

        #for test
    def transform(self, X_test):
        """
        Transform new data using the already fitted vectorizer.
        """
        if not self.is_fitted:
            raise ValueError("TfidfProcessor is not fitted yet. Call fit_transform() first.")
        return self.vectorizer.transform(X_test)
    
    def process(self):
        """
        Get feature names (words) used in the TF-IDF model.
        """
        x_train_tfidf =self.tfidf.fit_transform(self.train)
        x_test_tfidf = self.tfidf.transform(self.test)

        return  x_train_tfidf ,x_test_tfidf

