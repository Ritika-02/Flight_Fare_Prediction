import os
import sys
import numpy as np
import pandas as pd
from src.Flight_Fare_Prediction.logger import logging
from src.Flight_Fare_Prediction.exception import customexception
from src.Flight_Fare_Prediction.utils.utils import load_object


class PredictPipeline:
    # By default class , Construction class
    def __init__(self):
        pass


    def predict(self,features):
        try:
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
            return pred


        except Exception as e:
            raise customexception(e,sys)

'''one_hot_encode_cols = ['Airline', 'Source', 'Destination']
ordinal_encode_cols = ['Total_Stops']
numerical_cols = ['Duration_Hour','Duration_Min','Departure_Hour', 'Departure_Min', 'Arrival_Hour', 'Arrival_Min', 'Journey_Day', 'Journey_Month']'''

class CustomData:
    def __init__(self,
                 Airline:str,
                 Source: str,
                 Destination: str,
                 Journey_Day: int,
                 Journey_Month: int,
                 Total_Stops: str,
                 Duration_Hour: int,
                 Duration_Min: int,
                 Departure_Hour: int,
                 Departure_Min: int,
                 Arrival_Hour: int,
                 Arrival_Min: int):
        
        self.Airline = Airline
        self.Source = Source
        self.Destination = Destination
        self.Journey_Day = Journey_Day
        self.Journey_Month = Journey_Month
        self.Total_Stops = Total_Stops
        self.Duration_Hour = Duration_Hour
        self.Duration_Min = Duration_Min
        self.Departure_Hour = Departure_Hour
        self.Departure_Min = Departure_Min
        self.Arrival_Hour = Arrival_Hour
        self.Arrival_Min = Arrival_Min


    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Airline' :[self.Airline],
                'Source' :[self.Source],
                'Destination' : [self.Destination],
                'Journey_Day': [self.Journey_Day],
                'Journey_Month': [self.Journey_Month],
                'Total_Stops':[self.Total_Stops],
                'Duration_Hour':[self.Duration_Hour],
                'Duration_Min':[self.Duration_Min],
                'Departure_Hour':[self.Departure_Hour],
                'Departure_Min':[self.Departure_Min],
                'Arrival_Hour': [self.Arrival_Hour],
                'Arrival_Min':[self.Arrival_Min]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df

        except Exception as e:
            logging.info('Exception Occured in Prediction Pipeline')
            raise customexception(e,sys)
        