import os
import sys
import numpy as np
import pandas as pd 

from dataclasses import dataclass
from src.Flight_Fare_Prediction.exception import customexception
from src.Flight_Fare_Prediction.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.Flight_Fare_Prediction.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    

    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation Initiated')

            one_hot_encode_cols = ['Airline', 'Source', 'Destination']
            ordinal_encode_cols = ['Total_Stops']
            numerical_cols = ['Duration_Hour','Duration_Min','Departure_Hour', 'Departure_Min', 'Arrival_Hour', 'Arrival_Min', 'Journey_Day', 'Journey_Month']

            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )


            # Categorical Pipelines
            # One-Hot Encoding Pipeline
            one_hot_encode_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            # Ordinal Encoding Pipeline
            ordinal_encode_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[['non-stop','1 stop', '2 stops', '3 stops', '4 stops']]))
                ]
            )


            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('one_hot_encode', one_hot_encode_pipeline, one_hot_encode_cols),
                    ('ordinal_encode', ordinal_encode_pipeline, ordinal_encode_cols)
                ],
                remainder= 'passthrough'
            )

            return preprocessor


        except Exception as e:
            logging.info('Exception occured in initiate_datatransformation')

            raise customexception(e,sys)

    
    def initialize_data_transformation(self,train_path,test_path):
        
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data complete')
            logging.info(f'Train Dataframe Head : \n {train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n {test_df.head().to_string()}')

            for df in [train_df, test_df]:
                # Extracting features from 'Dep_Time'
                df['Departure_Hour'] = pd.to_datetime(df['Dep_Time'], format= "%H:%M").dt.hour
                df['Departure_Min'] = pd.to_datetime(df['Dep_Time'], format= "%H:%M").dt.minute
                
                # Extracting features from 'Arrival_Time'
                df['Arrival_Hour'] = df['Arrival_Time'].apply(lambda x : pd.to_datetime(x, errors= 'coerce').hour)
                df['Arrival_Min'] = df['Arrival_Time'].apply(lambda x : pd.to_datetime(x, errors= 'coerce').minute)

                # Extracting features from 'Date_of_Journey'
                df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey'], format= '%d/%m/%Y').dt.day
                df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey'], format= '%d/%m/%Y').dt.month

                # Extracting features from 'Date_of_Journey'
                df['Duration_Hour'] = df['Duration'].str.extract(r'(\d+)h',expand= False).fillna(0).astype(int)
                df['Duration_Min'] = df['Duration'].str.extract(r'(\d+)h',expand= False).fillna(0).astype(int)


            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'Price'
            drop_columns = [target_column_name,'Date_of_Journey', 'Route', 'Dep_Time','Arrival_Time', 'Duration','Additional_Info']

            input_feature_train_df = train_df.drop(columns = drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('Applying preprocessing object on training and testing datasets')
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info('preprocessing pickle file saved')

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info('Exception occured in initiate_datatransformation')

            raise customexception(e,sys)