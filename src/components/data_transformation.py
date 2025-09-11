import sys
import os
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )

            preproccessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_features),
                    ("cat_pipeline", categorical_pipeline, categorical_features)
                ]
            )

            return preproccessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate(self, train_path, test_path):
        try:
            logging.info("Data Transformation initiated")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")
            preprocessor = self.get_data_transformer_object()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis = 1)
            output_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis = 1)
            output_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            train_input_transformed_data = preprocessor.fit_transform(input_feature_train_df)
            test_input_transformed_data = preprocessor.transform(input_feature_test_df)

            train_data = np.c_[train_input_transformed_data, np.array(output_feature_train_df)]
            test_data = np.c_[test_input_transformed_data, np.array(output_feature_test_df)]

            save_obj(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            logging.info("Saved preprocessing object")

            return (
                train_data,
                test_data,
                self.transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)

