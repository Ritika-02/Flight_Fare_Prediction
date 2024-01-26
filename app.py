from src.Flight_Fare_Prediction.pipelines.prediction_pipeline import CustomData,PredictPipeline
from flask import Flask,request,render_template,jsonify
from src.Flight_Fare_Prediction.logger import logging


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")
#


@app.route('/predict', methods =["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data = CustomData(

            Airline= request.form.get('Airline'),
            Journey_Day= int(request.form.get('Journey_Day')),
            Journey_Month= int(request.form.get('Journey_Month')),
            Source= request.form.get('Source'),
            Destination= request.form.get('Destination'),
            Departure_Hour= int(request.form.get('Departure_Hour')),
            Departure_Min= int(request.form.get('Departure_Min')),
            Arrival_Hour= int(request.form.get('Arrival_Hour')),
            Arrival_Min= int(request.form.get('Arrival_Min')),
            Duration_Hour= int(request.form.get('Duration_Hour')),
            Duration_Min= int(request.form.get('Duration_Min')),
            Total_Stops= request.form.get('Total_Stops')
        )

        final_data = data.get_data_as_dataframe()
        logging.info(f'{final_data}')

        predict_pipeline = PredictPipeline()

        pred = predict_pipeline.predict(final_data)

        # Rounding the prediction value upto 2 points
        result = round(pred[0],2)

        return render_template("result.html",final_result = result)



if __name__ == '__main__':
    app.run()