import sys
from dataclasses import dataclass

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    k_best_selector_path=os.path.join('artifacts',"k_best_selector.pkl")
    rfe_selector_path=os.path.join('artifacts',"rfe_selector.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:

            # Create pipeline for feature selection (SelectKBest)
            k_best_selector_pipeline = Pipeline([
                ("selector", SelectKBest(score_func=f_classif, k=200))  # Set k as desired
            ])

            # Create pipeline for feature selection (RFE)
            estimator = RandomForestClassifier()
            k=100
            rfe_selector_pipeline = RFE(estimator,n_features_to_select=k)
            # rfe_selector_pipeline = Pipeline([
            #     ("estimator", estimator),  # You can use any estimator here
            #     ("selector", RFE(estimator, n_features_to_select=100))  # Set k as desired
            # ])
            
            
            categorical_columns = [
                "Activity"
            ]
            # Define pipeline with LabelEncoder
            label_encoder_pipeline = LabelEncoder()

            logging.info(f"Categorical columns: {categorical_columns}")
  

            return k_best_selector_pipeline, rfe_selector_pipeline, label_encoder_pipeline
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            duplicated_columns = ['tBodyAccMag-sma()',
            'tGravityAccMag-mean()',
            'tGravityAccMag-std()',
            'tGravityAccMag-mad()',
            'tGravityAccMag-max()',
            'tGravityAccMag-min()',
            'tGravityAccMag-sma()',
            'tGravityAccMag-energy()',
            'tGravityAccMag-iqr()',
            'tGravityAccMag-entropy()',
            'tGravityAccMag-arCoeff()1',
            'tGravityAccMag-arCoeff()2',
            'tGravityAccMag-arCoeff()3',
            'tGravityAccMag-arCoeff()4',
            'tBodyAccJerkMag-sma()',
            'tBodyGyroMag-sma()',
            'tBodyGyroJerkMag-sma()',
            'fBodyAccMag-sma()',
            'fBodyBodyAccJerkMag-sma()',
            'fBodyBodyGyroMag-sma()',
            'fBodyBodyGyroJerkMag-sma()']
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            k_best_selector_pipeline_obj, rfe_selector_pipeline_obj, label_encoder_pipeline_obj=self.get_data_transformer_object()

            target_column_name="Activity"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            

            input_train_num = input_feature_train_df.drop(duplicated_columns,axis=1)
            input_test_num = input_feature_test_df.drop(duplicated_columns,axis=1)
            input_train_k=k_best_selector_pipeline_obj.fit_transform(input_train_num,target_feature_train_df)
            input_test_k=k_best_selector_pipeline_obj.transform(input_test_num)
            input_train_rfe=rfe_selector_pipeline_obj.fit_transform(input_train_k,target_feature_train_df)
            input_test_rfe=rfe_selector_pipeline_obj.transform(input_test_k)


            target_train_label=label_encoder_pipeline_obj.fit_transform(target_feature_train_df)
            target_test_label=label_encoder_pipeline_obj.transform(target_feature_test_df)

            train_arr = np.c_[
                input_train_rfe, np.array(target_train_label)
            ]
            test_arr = np.c_[input_test_rfe, np.array(target_test_label)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.k_best_selector_path,
                obj=k_best_selector_pipeline_obj

            )
            save_object(

                file_path=self.data_transformation_config.rfe_selector_path,
                obj=rfe_selector_pipeline_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.k_best_selector_path,
                self.data_transformation_config.rfe_selector_path,
            )
        except Exception as e:
            raise CustomException(e,sys)