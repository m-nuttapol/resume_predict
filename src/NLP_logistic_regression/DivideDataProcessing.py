from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataDivide:
    def __init__(self, df):
        """Initialize the class with the DataFrame."""
        self.df = df
        self.le = LabelEncoder()

    def encode_categories(self):
        """Encode job categories into numerical values."""
        #transfrom to encode and 
        self.df['Category_Encoded'] = self.le.fit_transform(self.df['Category'])
        return self.df

    #every time you run the code, you get the same train-test split.
    #for now we have to train X (info data) and Y is label we train tgt cause if  X is match 
    #x and y have to equal lengths

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and test sets."""

        #stratify=y ensures that the proportion of each class in the training and test sets remains the same as in the original dataset.

        x = self.df['Cleaned_Resume']
        y = self.df['Category_Encoded']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def process(self):
        self.encode_categories()
        x_train, x_test, y_train, y_test = self.split_data()
        return x_train, x_test, y_train, y_test
