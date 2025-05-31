import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import numpy as np
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('arifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting train and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Setting up RandomForestRegressor and hyperparameter tuning")
            rf = RandomForestRegressor(random_state=2)
            param_dist = {
                'max_depth': list(range(2, 8)),
                'n_estimators': list(range(50, 150)),
                'max_features': ['auto', 'sqrt', 'log2']
            }

            rcv = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_dist,
                scoring='neg_mean_squared_error',
                cv=4,
                n_iter=10,
                random_state=42,
                verbose=2,
                n_jobs=-1
            )
            logging.info("Training model using RandomizedSearchCV")
            rcv.fit(X_train, y_train)

            best_model = rcv.best_estimator_
            logging.info(f"Best parameters: {rcv.best_params_}")

            logging.info("Evaluating model on test set")
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Test MSE: {mse}")
            logging.info(f"Test R² Score: {r2}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return mse, r2, self.model_trainer_config.trained_model_file_path

       
        except Exception as e:
            raise CustomException(e, sys)
