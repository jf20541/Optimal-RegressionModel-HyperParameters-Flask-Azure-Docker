
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import pandas as pd 
import joblib

# initialize the Flask app 
app = Flask(__name__)
# read the XGBoost model in read mode 
model = pickle.load(open('models/model.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    prediction = model.predict(int_features)
    return render_template(f'../templates/home.html', prediction_text="Home Prediction {prediction[0]}")
    
    
if __name__ == '__main__':
    app.run(debug=True)
    
    
    
# app = Flask(__name__)
# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict',methods = ['POST'])
# def predict():
#     model = joblib.load('models/model.pkl')
#     json = request.get_json()
#     temp=np.array(list(json[0].values()))
#     prediction = model.predict(temp)
#     return render_template(f'../templates/home.html', prediction_text="Home Prediction {prediction}")

# if __name__ == '__main__':
#     app.run(debug=True)