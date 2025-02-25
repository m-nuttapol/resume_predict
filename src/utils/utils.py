import pandas as pd
import os
def load_csv(file_path):
    """Load dataset from CSV"""
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        df = None
    return df


def save_df_to_csv(df, filename="clean_data.csv", index=False, encoding="utf-8"):
    """
    Saves a pandas DataFrame to a CSV file dynamically.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    filename (str): The name of the output CSV file (default: "clean_data.csv").
    index (bool): Whether to include the index column (default: False).
    encoding (str): Encoding format (default: "utf-8").

    Returns:
    None
    """
    try:
        # Get the absolute path of the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Corrected: Pass filename as a separate argument
        file_path = os.path.abspath(os.path.join(script_dir, "../../src/dataset", filename))
        df.to_csv(file_path, index=index, encoding=encoding)
        print(f"CSV file '{filename}' saved successfully!")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
