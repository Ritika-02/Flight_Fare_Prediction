import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.Flight_Fare_Prediction.logger import logging
from src.Flight_Fare_Prediction.exception import customexception
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import StackingRegressor

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            #train model
            model.fit(X_train,y_train)


            #predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 score for train and test data
            # train_model_score = r2_score(y_train,y_train_pred)
            test_model_Score = r2_score(y_test,y_test_pred)


            report[list(models.keys())[i]] = test_model_Score

        return report
   

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        logging.info('Exception occured in load_object function utils')
        raise customexception(e,sys)