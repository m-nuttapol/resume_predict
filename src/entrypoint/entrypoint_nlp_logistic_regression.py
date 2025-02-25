import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "NLP_logistic_regression")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))


from CheckDataBalance import DataBalance
from DivideDataProcessing import DataDivide
from TextRepresentation import Tfidf
from utils import load_csv






if __name__ == "__main__":


    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../../src/dataset/clean_data.csv")


    #Step 1. Load csv that clean data convert to DF
    clean_df = load_csv(file_path)



    #Step 2. Check databalance on data
    #rate_bal = if >5X will imbalance
    #rate_bal_percen -> 5 % that means data still ok 
    data_bal_proc = DataBalance(clean_df)
    rate_bal,rate_bal_percen  = data_bal_proc.process() 
    print(rate_bal)
    print(rate_bal_percen.tail(1))

    #Step 3. divide train and test data 

    data_div_proc = DataDivide(clean_df)
    x_train, x_test, y_train, y_test  = data_div_proc.process() 

    #Step 3. Apply tfidf (is a technique used in Natural Language Processing (NLP) to convert text into numerical values)
    # fit() → Learns the vocabulary and IDF values from the training data.
    # transform() → Converts the text into a TF-IDF numerical matrix using the learned vocabulary.
    # fit_transform() → Combines both steps (fit + transform) in one function.

    tfidf_proc = Tfidf(x_train,x_test)
    tfidt_x_train , tfidf_x_test = tfidf_proc.process()
    

    print("====")
    

    # if df is not None:
    #     # ✅ Print a sample cleaned resume
    #     print("\n📄 Cleaned Resume Sample:")
    #     print(df)

    # else:
    #     print("❌ No cleaned resume data available.")

