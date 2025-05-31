import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = r'src\components\arifacts\model.pkl'
            preprocessor_path = r'src\components\arifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
        


class CustomData:
    def __init__(self, Engine_Size: float,
                Cylinders: int, Fuel_Cons_100km: float,
                Fuel_Cons_mpg: int, Vehicle_Class: str, Fuel_Type: str):
        self.Engine_Size = Engine_Size
        self.Cylinders = Cylinders
        self.Fuel_Cons_100km = Fuel_Cons_100km
        self.Fuel_Cons_mpg = Fuel_Cons_mpg
        self.Vehicle_Class = Vehicle_Class
        self.Fuel_Type = Fuel_Type

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Engine Size(L)": [self.Engine_Size],
                "Cylinders": [self.Cylinders],
                "Fuel Consumption Comb (L/100 km)": [self.Fuel_Cons_100km],
                "Fuel Consumption Comb (mpg)": [self.Fuel_Cons_mpg],
                "Vehicle Class": [self.Vehicle_Class],
                "Fuel Type": [self.Fuel_Type],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)