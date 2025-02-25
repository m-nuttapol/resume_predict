import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from data_preprocessing import DataCleaning


if __name__ == "__main__":
    # ✅ Initialize and run processing in one step
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../../src/dataset/Resume_dataset.csv")

    data_cleaning_proc = DataCleaning(file_path)
    data_cleaning_proc.process()
   