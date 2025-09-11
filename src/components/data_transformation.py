import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_path
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.utils import save_path



class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('aritfacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''this function is responsible for data transformation'''

        try:
            numerical_columns=['reading score', 'writing score']
            categorical_columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),   
                    ("scaler",StandardScaler())
                ]
            )
        
            cat_pipepline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder",OneHotEncoder())
                ]
            )

            logging.info("Categorical Encoding Completed")
            logging.info("Numerical Scaling Completed")


            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipline",cat_pipepline,categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data")
            logging.info("Obtaining prepocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            target_column='math score'
            numerical_columns=['writing score','reading score']

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]
            
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info(f"Appyling preprocessing object on training dataframe and testing dataframe")


            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_ar=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_ar=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            

            logging.info("saved preprocessing object")

            save_path(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            ) 
            return(
                train_ar,test_ar,self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        






