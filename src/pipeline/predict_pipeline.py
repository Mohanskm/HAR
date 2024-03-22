import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
from PIL import Image


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
            image = {
                'Standing':'images/standing.jpeg',
                'Sitting':'images/standing.jpeg',
                'Laying':'images/standing.jpeg',
                'Walking_downstairs':'images/standing.jpeg',
                'Walking_upstairs':'images/standing.jpeg',
                'Walking':'images/standing.jpeg'
            }
            preds=model.predict(data_scaled)
            mapped_preds = [activity_labels[pred] for pred in preds]
            mapped_image = [image[i] for i in mapped_preds]
            return mapped_preds, mapped_image
        
            # for _, row in data_scaled.iterrows():
            #         preds_list=[]
            #         row_array = row.values.reshape(1, -1)  # Reshape the row to match the expected input format
            #         pred = model.predict(row_array)
            #         mapped_pred = activity_labels[pred[0]]  # Map numerical prediction to activity label
            #         preds_list.append(mapped_pred)

            #         return preds_list
        
        except Exception as e:
            raise CustomException(e,sys)


    def predict_and_save(self, csv_file_path):
        try:
            custom_data = CustomData(csv_file_path)
            data_frame = custom_data.get_data_as_data_frame()
            predictions = self.predict(data_frame)
            predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
            output_df = pd.concat([data_frame, predictions_df], axis=1)
            # Convert DataFrame to CSV string
            csv_string = output_df.to_csv(index=False)
             # Save the CSV string to a file
            csv_filename = 'predictions.csv'
            with open(csv_filename, 'w') as csv_file:
                csv_file.write(csv_string)
            # Save the image of the pie chart
            image = custom_data.plot_pie_chart(predictions)
            # image_filename = 'pie_chart.png'
            # image.save(image_filename)
            # Return the CSV string and predictions for handling in the route
            return image
        except Exception as e:
            raise CustomException(e, sys)

            


class CustomData:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def get_data_as_data_frame(self):
        try:
            df = pd.read_csv(self.csv_file_path)
            return df
        except Exception as e:
            raise CustomException(e, sys)
    
    def plot_pie_chart(self, y_pred_labels):
        label_counts = pd.Series(y_pred_labels).value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Predicted Activity Distribution')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Save the plot to an image buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()  # Close the figure to release memory
        
        # Create a PIL image object from the byte buffer
        img_buffer.seek(0)
        pil_image = Image.open(img_buffer)
        pie = pil_image.show()
        return pie
        # # Save the plot to an image file
        # img_file = 'pie_chart.png'
        # plt.savefig(img_file)
        # plt.close()  # Close the figure to release memory
        
        # return img_file
# if __name__=="__main__":

#     # custom_data = CustomData(csv_file_path="One_point.csv")
#     # data_frame = custom_data.get_data_as_data_frame()
#     obj=PredictPipeline()
#     image_path = obj.predict_and_save("new_test.csv")
#     pie = image_path.show()
#     print(pie)