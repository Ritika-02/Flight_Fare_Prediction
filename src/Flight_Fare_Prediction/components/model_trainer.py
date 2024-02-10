import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.Flight_Fare_Prediction.logger import logging
from src.Flight_Fare_Prediction.exception import customexception
from src.Flight_Fare_Prediction.utils.utils import save_object
from src.Flight_Fare_Prediction.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    

    def initiate_model_training(self,train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (

                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            

            # Hyperparameter tuning using RandomSearchCV for RandomForestRegressor
            param_grid_rf = {
                'n_estimators': [100, 200, 300, 400,500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8]
            }

            rf_random = RandomizedSearchCV(RandomForestRegressor(), param_grid_rf, cv=5, scoring='r2', n_jobs= -1, random_state= 42)
            rf_random.fit(X_train, y_train)

            best_rf_model = rf_random.best_estimator_

            param_grid_xgb = {
                'learning_rate' : [0.01, 0.1, 0.2],
                'max_depth' : [3, 5, 7],
                'n_estimators' : [50, 100, 200]
            }

            xgb_random = RandomizedSearchCV(XGBRegressor(), param_grid_xgb, cv=5, scoring= 'r2', n_jobs= -1, random_state= 42)
            xgb_random.fit(X_train,y_train)

            best_xgb_model = xgb_random.best_estimator_

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso'            : Lasso(max_iter=10000),
                'Ridge'             : Ridge(),
                'DecisionTreeRegressor'  : DecisionTreeRegressor(),
                'RandomForestRegressor'  : best_rf_model,
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'XGBoostRegressor': best_xgb_model,
                'AdaBoostRegressor' : AdaBoostRegressor(best_rf_model, n_estimators=50, random_state= 42),
                'BaggingRegressor': BaggingRegressor(best_rf_model, n_estimators=50, random_state=42)
                }



            # Evaluate the best RandomForestRegressor model on the test set
            y_pred_rf = best_rf_model.predict(X_test)
            r2_rf = r2_score(y_test, y_pred_rf)


            print(f'Best RandomForestRegressor Model Found, R2 Score: {r2_rf}')
            print('Best Hyperparameters:', rf_random.best_params_)
            print('\n==================================================================================\n')
            logging.info(f'Best RandomForestRegressor Model Found, R2 Score: {r2_rf}')

            model_report : dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n==============================================================\n')
            logging.info(f'Model Report : {model_report}')


            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}')
            print('\n==================================================================================\n')
            logging.info(f'Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}')


            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)