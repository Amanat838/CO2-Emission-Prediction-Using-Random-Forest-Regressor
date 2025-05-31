import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DatTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('arifacts', 'train.csv')
    test_data_path = os.path.join('arifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion process')
        try:
            df = pd.read_csv(r'D:\Development_Content\MACHINE_LEARNING\CO2_Emission_Vehicles\notebook\data\CO2 Emissions_Canada.csv')
            logging.info('Read the dataset as data frame')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info('Train Test Split Initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            numerical_cols_train = train_set.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cols_to_drop = ['Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)']
            numerical_cols_train = [col for col in numerical_cols_train if col not in cols_to_drop]
            cat_cols_train = train_set.select_dtypes(include=['object', 'category']).columns
            selected_cat_cols_train = [col for col in cat_cols_train if df[col].nunique() <= 20]
            final_train_data = df[numerical_cols_train + selected_cat_cols_train]
            final_train_data = final_train_data.drop_duplicates()
            final_train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            numerical_cols_test = test_set.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cols_to_drop1 = ['Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)']
            numerical_cols_test = [col for col in numerical_cols_test if col not in cols_to_drop1]
            cat_cols_test = test_set.select_dtypes(include=['object', 'category']).columns
            selected_cat_cols_test = [col for col in cat_cols_test if df[col].nunique() <= 20]
            final_test_data = df[numerical_cols_test + selected_cat_cols_test]
            final_test_data = final_test_data.drop_duplicates()
            final_test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data Ingestion Completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path 
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
