# handler.py

from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import os

app = Flask(__name__,static_url_path='/static')



# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting data
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('home.html', error='No file part')

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return render_template('home.html', error='No file selected')

        # Check if file has allowed extension
        if not file.filename.endswith('.csv'):
            return render_template('home.html', error='File format not supported. Please upload a CSV file.')

        # Process the CSV file
        custom_data = CustomData(csv_file_path=file)
        data_frame = custom_data.get_data_as_data_frame()

        # Perform prediction
        predict_pipeline = PredictPipeline()
        results= predict_pipeline.predict(data_frame)

        return render_template('home.html', results=results[0])
    

# Route for uploading and predicting multiple data points
@app.route('/predict_and_save', methods=['GET', 'POST'])
def predict_and_save():
    if request.method == 'GET':
        return render_template('multi.html')
    else:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('multi.html', error='No file part')

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return render_template('multi.html', error='No file selected')

        # Check if file has allowed extension
        if not file.filename.endswith('.csv'):
            return render_template('multi.html', error='File format not supported. Please upload a CSV file.')

        # Perform prediction and save results
        predict_pipeline = PredictPipeline()
        pie_chart_img_path = predict_pipeline.predict_and_save(file)

        return render_template('multi.html', pie_chart=pie_chart_img_path)
        


# if __name__ == "__main__":
#     app.run(host="0.0.0.0")
