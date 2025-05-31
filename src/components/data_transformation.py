import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os

# Used to make pipelines for different columns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # Used to fill missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
# This is a custom utility function to save the preprocessor object
from src.utils import save_object


class DatTransformationConfig:
    preprocessor_obj_file_path = os.path.join('arifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DatTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Engine Size(L)',
                                'Cylinders',
                                'Fuel Consumption Comb (L/100 km)',
                                'Fuel Consumption Comb (mpg)']
            categorical_columns = ['Vehicle Class', 'Fuel Type']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(sparse_output=False))
            ])

            logging.info('Numerical and Categorical Pipelines Created')
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data as dataframes")
            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()
            target_feature_name = 'CO2 Emissions(g/km)'
            input_feature_train_df = train_df.drop(columns=[target_feature_name], axis=1)
            target_feature_train_df = train_df[target_feature_name]
            input_feature_test_df = test_df.drop(columns=[target_feature_name],axis=1)
            target_feature_test_df = test_df[target_feature_name]
            logging.info("Applying preprocessing object on training and testing dataframes") 
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info(f"Input Feature Train Shape: {input_feature_train_arr.shape}")
            logging.info(f"Target Feature Train Shape: {np.array(target_feature_train_df).shape}")


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor_obj)


            return (
                train_arr,
                test_arr,
            )
        
        except Exception as e:
            raise CustomException(e, sys)