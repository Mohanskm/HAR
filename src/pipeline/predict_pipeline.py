import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
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
            model_path=os.path.join("artifacts","model.pkl")
            k_best_selector_path=os.path.join('artifacts','k_best_selector.pkl')
            rfe_selector_path=os.path.join('artifacts','rfe_selector.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            k_best_selector=load_object(file_path=k_best_selector_path)
            rfe_selector=load_object(file_path=rfe_selector_path)
            print("After Loading")
            input_train = features.drop(duplicated_columns,axis=1)
            k_data_scaled=k_best_selector.transform(input_train)
            data_scaled=rfe_selector.transform(k_data_scaled)

            # row_to_predict = data_scaled.iloc[1:2]
            
            # Map numerical predictions to activity labels
            activity_labels = {
                0: 'Standing',
                1: 'Sitting',
                2: 'Laying',
                3: 'Walking_downstairs',
                4: 'Walking_upstairs',
                5: 'Walking'
            }
            preds=model.predict(data_scaled)
            mapped_preds = [activity_labels[pred] for pred in preds]
            return mapped_preds
        
            # for _, row in data_scaled.iterrows():
            #         preds_list=[]
            #         row_array = row.values.reshape(1, -1)  # Reshape the row to match the expected input format
            #         pred = model.predict(row_array)
            #         mapped_pred = activity_labels[pred[0]]  # Map numerical prediction to activity label
            #         preds_list.append(mapped_pred)

            #         return preds_list
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def get_data_as_data_frame(self):
        try:
            df = pd.read_csv(self.csv_file_path)
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
# if __name__=="__main__":

#     custom_data = CustomData(csv_file_path="One_point.csv")
#     data_frame = custom_data.get_data_as_data_frame()
#     obj=PredictPipeline()
#     print(obj.predict(data_frame))