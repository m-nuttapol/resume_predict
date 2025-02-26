

from imblearn.over_sampling import SMOTE

class HandleBalance:
    def __init__(self, tfidt_x_train,y_train):
        self.tfidt_x_train = tfidt_x_train
        self.y_train = y_train
 
    def apply_smote(self):
        """Apply SMOTE to handle class imbalance."""
        smote = SMOTE(sampling_strategy="auto", k_neighbors=3, random_state=42)  # Reduce neighbors
        X_resampled, y_resampled = smote.fit_resample(self.tfidt_x_train, self.y_train)

        print("✅ After SMOTE:")
        print("X_resampled shape:", X_resampled.shape)
        print("y_resampled shape:", y_resampled.shape)

        return X_resampled, y_resampled

    
    def process(self):
        return self.apply_smote()




