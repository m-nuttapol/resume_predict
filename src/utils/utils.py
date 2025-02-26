import pandas as pd
import os
import fitz
import json

def load_csv(file_path):
    """Load dataset from CSV"""
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        df = None
    return df


def save_df_to_csv(df, filename="cleaned_data.csv", index=False, encoding="utf-8"):
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
        file_path = os.path.abspath(os.path.join(script_dir, "../../src/clean_dataset", filename))
        df.to_csv(file_path, index=index, encoding=encoding)
        print(f"CSV file '{filename}' saved successfully!")
    except Exception as e:
        print(f"Error saving CSV file: {e}")



def save_df_to_excel(df, file_name="cleaned_data.xlsx"):
    """
    Saves a pandas DataFrame to a Excel file dynamically.

    Parameters:
    filename (str): The name of the output excel file (default: "clean_data.xlsx").
    index (bool): Whether to include the index column (default: False).

    Returns:
    None
    """
    try:
        # Get the absolute path of the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))


        # Corrected: Pass filename as a separate argument
        file_path = os.path.abspath(os.path.join(script_dir, "../../src/clean_dataset", file_name))
        df.to_excel(file_path, index=False)
        print(f"CSV file '{file_name}' saved successfully!")
    except Exception as e:
        print(f"Error saving CSV file: {e}")



def read_pdfs_from_folder(root_folder):
    """
    Reads all PDFs from the given folder (including subfolders) and extracts text.

    Args:
        root_folder (str): Path to the dataset folder.

    Returns:
        list: A list of dictionaries containing job role, file name, and extracted text.
    """
    resume_data = []

    for job_role in os.listdir(root_folder):
        job_role_path = os.path.join(root_folder, job_role)

        if os.path.isdir(job_role_path):  # Check if it's a folder
            print(f"📂 Processing job role: {job_role}")

            for file_name in os.listdir(job_role_path):
                if file_name.endswith(".pdf"):  # Process only PDFs
                    pdf_path = os.path.join(job_role_path, file_name)

                    try:
                        doc = fitz.open(pdf_path)  # ✅ Correct way to open PDF
                        text = "\n".join(page.get_text("text") for page in doc)  # ✅ Correct way to iterate

                        resume_data.append({
                            "Job Role": job_role,
                            "File Name": file_name,
                            "Extracted Text": text
                        })

                    except Exception as e:
                        print(f"❌ Error reading {file_name}: {e}")

    return resume_data



def load_skills_from_json(file_path):
    """
    Loads skills from a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: Dictionary of job categories and their associated skills.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        skills_dict = json.load(file)
    
    # Flatten all skills into a single list
    all_skills = set(skill for skills in skills_dict.values() for skill in skills)
    
    return skills_dict, list(all_skills)



def load_csvs_from_folder(folder_path):
    """
    Reads all CSV files in the specified folder and returns a dictionary of DataFrames.
    
    Args:
        folder_path (str): The path to the folder containing CSV files.
    
    Returns:
        dict: A dictionary where keys are filenames (without extensions) and values are DataFrames.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Dictionary to store DataFrames
    dfs = {}

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df_name = file.replace(".csv", "")  # Use filename (without extension) as key
        dfs[df_name] = pd.read_csv(file_path)

    return dfs


def merge_similar_dataframes(dfs):
    """
    Merges all DataFrames that have the same column names and data types.

    Args:
        dfs (dict): Dictionary of DataFrames.

    Returns:
        tuple:
            - merged_df (pd.DataFrame): The merged DataFrame of compatible CSVs.
            - merged_files (list): List of filenames that were successfully merged.
            - skipped_files (list): List of filenames that were skipped due to structure differences.
    """
    if not dfs:
        return None, [], []

    first_file = list(dfs.keys())[0]  # Take the first file as reference
    first_df = dfs[first_file]
    merged_df = first_df.copy()

    merged_files = [first_file]  # List to track merged files
    skipped_files = []  # List to track skipped files

    for name, df in list(dfs.items())[1:]:  # Compare remaining DataFrames
        if df.columns.tolist() == first_df.columns.tolist() and df.dtypes.tolist() == first_df.dtypes.tolist():
            merged_df = pd.concat([merged_df, df], ignore_index=True)  # Append if structure matches
            merged_files.append(name)
        else:
            skipped_files.append(name)

    return merged_df, merged_files, skipped_files



def apply_roles(df, role_mapping):
    """
    Maps job positions in the 'Category' column of the DataFrame to their respective roles
    based on the provided role mapping.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing a "Category" column with job positions.
    - role_mapping (dict): The role mapping dictionary where job positions map to roles.

    Returns:
    - pd.DataFrame: The modified DataFrame with a new column "map_roles".
    """
    if "Category" not in df.columns:
        raise ValueError("❌ The DataFrame must contain a 'Category' column!")

    # Map positions to roles
    df["map_roles"] = df["Category"].map(role_mapping)

    return df


def load_role_and_skills(json_file_path):
    """Loads role mapping from a JSON file and returns position-to-role mapping and a list of all lowercase skills."""
    try:
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            role_mapping = json.load(json_file)

        position_to_role = {}  # Maps positions to their respective roles
        all_skills = set()  # Set to collect all unique skills (in lowercase)

        # Extract positions and skills
        for category, subcategories in role_mapping["role"].items():
            for subcategory, details in subcategories.items():
                positions = details.get("positions", [])
                skills = details.get("skills", [])
                keywords = details.get("keywords", [])  # Get keywords if present
                
                # Append keywords to skills list and convert to lowercase
                combined_skills = {skill.lower() for skill in skills + keywords}

                # Map positions to their respective categories
                for position in positions:
                    position_to_role[position] = subcategory
                
                # Collect all lowercase skills into a single list
                all_skills.update(combined_skills)

        return position_to_role, list(all_skills)  # Convert skills to list for output
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}, []