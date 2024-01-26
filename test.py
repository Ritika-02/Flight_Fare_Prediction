from src.Flight_Fare_Prediction.pipelines.prediction_pipeline import CustomData

# "IndiGo","24/03/2019","Banglore","New Delhi","BLR → DEL","22:20","01:10 22 Mar","2h 50m","non-stop","No info","3897"
custdata =CustomData("IndiGo","Banglore","New Delhi",24,3,"non-stop",2,50,22,20,1,10)

data = custdata.get_data_as_dataframe()
print(data)