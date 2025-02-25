import matplotlib.pyplot as plt
import seaborn as sns

class DataBalance:
    def __init__(self, df):
        """
        Initializes the DataBalance class with a DataFrame containing job categories.
        """
        self.df = df
        self.category_counts = df['Category'].value_counts()  
        self.total_samples = len(df)

    def plot_distribution(self):
        """
        Plots the distribution of job categories.
        """
        plt.figure(figsize=(10, 5))
        sns.barplot(x=self.category_counts.index, y=self.category_counts.values)
        plt.xticks(rotation=90)
        plt.xlabel("Job Category")
        plt.ylabel("Count")
        plt.title("Category Distribution")
        plt.show()

    def calculate_proportion(self):
        """
        Calculates the proportion between the largest and smallest job category.
        Returns:
            - Proportion of largest category / smallest category
            - Percentage of each category
        """
        proportion_largest_smallest = self.category_counts.iloc[0] / self.category_counts.iloc[-1]

        category_percentages = (self.category_counts / self.total_samples) * 100

        return proportion_largest_smallest, category_percentages

    def process(self):
        """
        Runs full processing: plots category distribution & prints category proportions.
        """
        print("📊 Plotting category distribution...")
        self.plot_distribution()
        
        proportion_largest_smallest, category_percentages = self.calculate_proportion()
        return proportion_largest_smallest,category_percentages




