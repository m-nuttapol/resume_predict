import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from DataPreprocessing import DataCleaning
from EvaluateSkillSet import EvaluateSkill

from utils import load_csvs_from_folder,merge_similar_dataframes


if __name__ == "__main__":
    # ✅ Initialize and run processing in one step
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../../src/dataset/")
    
    dfs = load_csvs_from_folder(file_path)
    
    merged_df, merged_files, skipped_files = merge_similar_dataframes(dfs)

    # Print results
    print("✅ Merged DataFrame:")
    print(merged_df.head(), "\n")
    print(f"✅ Files merged: {merged_files}")
    print(f"⚠️ Files skipped due to different structures: {skipped_files}")

    data_cleaning_proc = DataCleaning(merged_df)
    clean_df = data_cleaning_proc.process()


    # load skill
    skill_counts = clean_df['Category'].explode().value_counts()
    print(skill_counts)




    config_file_path = os.path.join(script_dir, "../../src/config.json")

    eval_proc = EvaluateSkill(clean_df,config_file_path)
    clean_df_with_eval = eval_proc.process()


   