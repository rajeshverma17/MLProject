import sys
import os
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from src.utils import Evaluate_model

@dataclass
class ModelTrainerConfig:
    model_file_path=os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_file_path=ModelTrainerConfig()
    def ModelTrainer(self,train_data,test_data):
        try:
            logging.info("Model Training started")
            logging.info("Extract the data from train and test data")
            X_train,y_train,X_test,y_test=(
                train_data[:,0:-1],
                train_data[:,-1],
                test_data[:,0:-1],
                test_data[:,-1]
            )
            models={
               "Random Forest":RandomForestRegressor(),
               "Linear Regression":LinearRegression(),
               "XGRegressor":XGBRegressor(),
               "CatBoostRegressor":CatBoostRegressor(),
               "Adaboost Regressor":AdaBoostRegressor(),
               "DecisionTree Regressor":DecisionTreeRegressor()
            }
            report:dict= Evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_max_score=max(sorted(list(report.values())))
            best_model_name=list(report.keys())[
                list(report.values()).index(best_max_score)
            ]
            best_model=models[best_model_name]
            if best_max_score<=0.6:
                raise CustomException("No Best model is found")
            logging.info(f"best model is {best_model_name} and score is {best_max_score} and best model is {best_model}")
            y_predicted=best_model.predict(X_test)

            predicted_r2_score=r2_score(y_test,y_predicted)

            return predicted_r2_score

        except CustomException as e:
            raise CustomException(e,sys)
