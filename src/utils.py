import os
import dill
import sys
import pickle
from src.exception import  CustomException
from sklearn.metrics import r2_score
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as ex:
        raise CustomException(ex,sys)
    
def Evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            train_score=r2_score(y_train,y_train_pred)
            test_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_score   
            logging.info("--------------------------------------------------------------------------")
            logging.info(f"Model Name {list(models.keys())[i]} having R2 Score as : {test_score}")    
            logging.info("--------------------------------------------------------------------------")     
        return report
    except CustomException as e:
        raise CustomException(e,sys)
