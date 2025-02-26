import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import Counter

class DataTrainModel:
    def __init__(self, tfidt_x_train, tfidt_x_test, y_train, y_test, script_dir, model_name="resume_model.pkl"):
        self.tfidt_x_train = tfidt_x_train
        self.tfidt_x_test = tfidt_x_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.model_path = os.path.abspath(os.path.join(script_dir, "../../src/model", model_name))

        
    def train_model(self):
        """Train Logistic Regression model."""
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(self.tfidt_x_train, self.y_train)
        return model
    
    def test_model(self, model):
        """Test the model and generate classification report."""
        y_pred = model.predict(self.tfidt_x_test)  
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Extract weighted average F1-score for evaluation
        weighted_f1_score = report["weighted avg"]["f1-score"]

        # Calculate class proportions
        proportion_largest_smallest, category_percentages = self.calculate_proportion()

        return proportion_largest_smallest, category_percentages, weighted_f1_score, model, report

    def calculate_proportion(self):
        """Calculate the proportion of the largest to smallest class."""
        class_counts = Counter(self.y_train)
        largest_class = max(class_counts.values())
        smallest_class = min(class_counts.values())

        proportion_largest_smallest = largest_class / smallest_class if smallest_class > 0 else float('inf')
        category_percentages = {cls: round((count / sum(class_counts.values())) * 100, 2) for cls, count in class_counts.items()}

        print("Class Distribution:", class_counts)
        print(f"Proportion of largest to smallest class: {proportion_largest_smallest:.2f}")
        print("Class percentages:", category_percentages)

        return proportion_largest_smallest, category_percentages

    def save_model(self, model):
        """Save the trained model if F1-score meets the threshold."""
        joblib.dump(model, self.model_path)
        print(f"✅ Model saved at: {self.model_path}")

    def process(self, save_threshold=0.85):
        """Train, test, and save the model if performance is above the threshold."""
        print("✅ Training Logistic Regression...")
        print("X_train shape:", self.tfidt_x_train.shape)  
        print("y_train shape:", self.y_train.shape)

        model = self.train_model()
        proportion_largest_smallest, category_percentages, weighted_f1_score, model, report = self.test_model(model)

        # ✅ Save the model if weighted F1-score >= threshold
        if weighted_f1_score >= save_threshold:
            self.save_model(model)
        else:
            print(f"⚠️ Model NOT saved. Weighted F1-score ({weighted_f1_score:.2f}) is below {save_threshold}")

        return model, proportion_largest_smallest, category_percentages, weighted_f1_score, report
