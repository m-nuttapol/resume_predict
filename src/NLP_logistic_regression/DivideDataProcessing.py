from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import joblib
class DataDivide:
    def __init__(self, df,script_dir,label_encoder_name="label_encoder.pkl"):
        """Initialize the class with the DataFrame."""
        self.df = df
        self.le = LabelEncoder()

        self.model_path = os.path.abspath(os.path.join(script_dir, "../../src/model", label_encoder_name))


    def encode_categories(self):
        """Encode job categories into numerical values."""
        #transfrom to encode and 
        self.df['roles_encoded'] = self.le.fit_transform(self.df['map_roles'])

        # Save the LabelEncoder
        joblib.dump(self.le, self.model_path)

        print(f"✅ Label encoder saved at {self.model_path}")
        
        return self.df

    #every time you run the code, you get the same train-test split.
    #for now we have to train X (info data) and Y is label we train tgt cause if  X is match 
    #x and y have to equal lengths

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and test sets."""


        #stratify=y ensures that the proportion of each class in the training and test sets remains the same as in the original dataset.

        x = self.df[['cleaned_text', 'extract_skills']]  
        y = self.df['roles_encoded']


        # smote = SMOTE(sampling_strategy="auto", random_state=42)
        # X_resampled, y_resampled = smote.fit_resample(x, y)

        print("Original X shape:", x.shape)
        print("Original y shape:", y.shape)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)
        # x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size, stratify=y, random_state=random_state)


        print("✅ After Split")
        print("X_train shape:", x_train.shape)  
        print("X_test shape:", x_test.shape) 
        print("y_train shape:", y_train.shape) 
        print("y_test shape:", y_test.shape)
        
        return x_train, x_test, y_train, y_test

    def process(self):
        self.encode_categories()

        x_train, x_test, y_train, y_test = self.split_data()
        return x_train, x_test, y_train, y_test
