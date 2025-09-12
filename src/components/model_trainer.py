import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj, evaluate_models
from dataclasses import dataclass
import numpy as np
import pandas as pd

# models
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR 
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate(self, train_data, test_data):
        try:
            logging.info("Model Trainer initiated")

            logging.info("Splitting training and test input data")
            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "SVR": SVR(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            r2_list = []
            model_list = []

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)
            logging.info(f"Model report generated as : {model_report}")

            best_model_score = max(sorted(model_report.values()))
            #fetch the index of best model score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with r2 score: {best_model_score}")

            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info("Model saved successfully")

            pred = best_model.predict(X_test)
            r2_value = r2_score(y_test, pred)
            return r2_value

        except Exception as e:
            raise CustomException(e, sys)
