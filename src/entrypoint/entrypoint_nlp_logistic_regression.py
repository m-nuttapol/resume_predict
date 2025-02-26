import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "NLP_logistic_regression")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))


from CheckDataBalance import DataBalance
from DivideDataProcessing import DataDivide
from TextRepresentation import Tfidf
from HandleBalance import HandleBalance

from TrainModel import DataTrainModel
from utils import load_csv


pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_colwidth', None)  # Prevent text truncation


if __name__ == "__main__":


    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../../src/clean_dataset/cleaned_dataset.csv")
    

    # Step 1. Load csv that clean data convert to DF
    df = load_csv(file_path)

    #Step 2. Check databalance on data
    #rate_bal = if >5X will imbalance
    #rate_bal_percen -> 5 % that means data still ok 
    data_bal_proc = DataBalance(df)
    rate_bal,rate_bal_percen  = data_bal_proc.process() 
    print(rate_bal)
    print(rate_bal_percen.tail(1))

    #Step 3. divide train and test data 

    data_div_proc = DataDivide(df)
    x_train, x_test, y_train, y_test  = data_div_proc.process() 

    #Step 3. Apply tfidf (is a technique used in Natural Language Processing (NLP) to convert text into numerical values)
    # fit() → Learns the vocabulary and IDF values from the training data.
    # transform() → Converts the text into a TF-IDF numerical matrix using the learned vocabulary.
    # fit_transform() → Combines both steps (fit + transform) in one function.


    #Step 3. transfrom train x train data and x test to tfidf (for train do fit_transform but test just transform )
    tfidf_proc = Tfidf(x_train,x_test,script_dir=script_dir)
    tfidt_x_train , tfidf_x_test = tfidf_proc.process()

    handle_proc = HandleBalance(tfidt_x_train,y_train)
    tfidt_x_train , y_train = handle_proc.process()
    
    


    #Step 4. train model and test predict
    train_model_proc = DataTrainModel(tfidt_x_train,tfidf_x_test , y_train , y_test, script_dir)
    train_model_proc.process()
    print("====")
