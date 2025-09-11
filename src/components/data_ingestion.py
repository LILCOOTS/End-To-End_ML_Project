import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw_data.csv")
    train_data_path:str = os.path.join("artifacts", "train_data.csv")
    test_data_path:str = os.path.join("artifacts", "test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate(self):
        logging.info("Data Ingestion started")

        try:
            df = pd.read_csv("notebooks\data\stud.csv")
            logging.info("Dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, header=True, index = False)
            logging.info("Raw data saved")

            logging.info("Train test split initiated")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, header=True, index = False)
            logging.info("Train data saved")

            test_data.to_csv(self.ingestion_config.test_data_path, header=True, index = False)
            logging.info("Test data saved")

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate()