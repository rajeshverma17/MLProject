import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataInjectionClass:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")

class DataInjection:
    def __init__(self):
        self.injestion_config=DataInjectionClass()
    def initiate_data_injection(self):
        
        logging.info("Entered into data injestion method or component")
        try:
            df=pd.read_csv('notebook//data//stud.csv')     
            logging.info("Data has been read from  csv")
            os.makedirs(os.path.dirname(self.injestion_config.train_data_path),exist_ok=False)
            df.to_csv(self.injestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw data has been saved into csv")

            train_data,test_data=train_test_split(df,train_size=0.8,random_state=42)
            logging.info("Data set is splitted into train and test")
            
            df.to_csv(self.injestion_config.train_data_path,index=False,header=True)
            logging.info("Train data has been saved into csv")
            
            df.to_csv(self.injestion_config.test_data_path,index=False,header=True)
            logging.info("Test data has been saved into csv")

            return (
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj=DataInjection()
    obj.initiate_data_injection()