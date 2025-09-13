import sys
import os 
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor, RandomForestRegressor)

from sklearn.linear_model import   LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_path,evaluate_models




@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("aritfacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info("split training and testing input")

            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            

            models={
                "random forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Adaboost":AdaBoostRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "KNN":KNeighborsRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor()
            }

            params = {
                "random forest": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"]
                },

                "Decision Tree": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "sqrt", "log2"]
                },

                "Adaboost": {
                    "n_estimators": [50, 100, 200, 500],
                    "learning_rate": [0.001, 0.01, 0.1, 1]
                },

                "Gradient Boost": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 10],
                    "subsample": [0.6, 0.8, 1.0],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7, 11, 15],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]  # 1=manhattan, 2=euclidean
                },

                "XGBoost": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "gamma": [0, 0.1, 0.2],
                    "reg_alpha": [0, 0.01, 0.1],
                    "reg_lambda": [1, 1.5, 2]
                },

                "CatBoost": {
                    "iterations": [200, 500, 1000],
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "l2_leaf_reg": [1, 3, 5, 7],
                    "border_count": [32, 64, 128]
                }
            }


            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)


            best_model_score=max(model_report.values())

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("not best moedl found")
            
            logging.info(f"Best found model on both training and testing data")
            
            save_path(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)






























