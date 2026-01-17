import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.preprocessor_file_path=DataTransformationConfig.preprocessor_obj_file_path
    def get_data_transfer_object(self):
        try:
            num_features=['reading_score', 
                          'writing_score']
            cate_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
       'test_preparation_course']
            numerical_pipeline=Pipeline([
                ("Imputer",SimpleImputer(strategy="median")),
                ("StandardScaler",StandardScaler())
            ])
            categorical_pipeline=Pipeline(
                [
                    ("SimpleImputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder",OneHotEncoder()),
            ("StandardScalar",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical Features :{cate_features}")
            logging.info(f"Numerical Features {num_features}")
            preprocessor_transformer=ColumnTransformer([
                ("numerical_pipeline",numerical_pipeline,num_features),
                ("categorical_pipeline",categorical_pipeline,cate_features)
            ])
            return preprocessor_transformer
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading of test and train data complelted")
            logging.info("getting preprocessor object")
            preprocessor_obj=self.get_data_transfer_object()
            
            target_column_name='math_score'
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"applying processing object on train and test data frame")
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessor_obj.transform(input_feature_test_df)
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saving objects into pickle file")
            save_object(
                file_path=DataTransformationConfig.preprocessor_obj_file_path,
                obj=preprocessor_obj
                )
            return(
                train_arr,
                test_arr,
                self.preprocessor_file_path
            )
        except Exception as ex:
            raise CustomException(ex,sys)



        