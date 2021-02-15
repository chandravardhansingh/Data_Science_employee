import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('./test.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    '''
    city = request.form.get('city')
    city_development_index = request.form.get('city_development_index')
    gender = request.form.get('gender')
    relevent_experience = request.form.get('relevent_experience')
    enrolled_university = request.form.get('enrolled_university')
    education_level = request.form.get('education_level')
    major_discipline = request.form.get('major_discipline')
    experience = request.form.get('experience')
    company_size = request.form.get('company_size')
    company_type = request.form.get('company_type')
    last_new_job = request.form.get('last_new_job')
    training_hours = request.form.get('training_hours')

    features = [city,city_development_index,gender,relevent_experience,enrolled_university,education_level,major_discipline,experience,company_size,company_type,last_new_job,training_hours]

    return render_template('test.html', prediction_text='values: {}'.format(features))


if __name__ == "__main__":
    app.run(debug=True)