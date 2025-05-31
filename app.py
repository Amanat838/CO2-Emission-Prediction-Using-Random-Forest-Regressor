from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/makeprediction', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Engine_Size = request.form.get('engine_size'),
            Cylinders = request.form.get('cylinders'),
            Fuel_Cons_100km = request.form.get('fuel_cons_100km'),
            Fuel_Cons_mpg = request.form.get('fuel_cons_mpg'),
            Fuel_Type = request.form.get('fuel_type'),
            Vehicle_Class = request.form.get('vehicle_class')
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0")  