from flask import Flask, render_template, request
import pickle
import numpy as np
import time
from datetime import datetime


app = Flask(__name__)

model_path = "models/model.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['Name']
        address = request.form['Address']
        gender = int(request.form['Gender'])
        married = int(request.form['Married'])
        dependents = int(request.form['Dependents'])
        education = int(request.form['Education'])
        self_employed = int(request.form['Self_Employed'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = int(request.form['Credit_History'])
        property_area = int(request.form['Property_Area'])
        total_income = float(request.form['Total_Income'])

        # Model input
        input_features = np.array([[gender, married, dependents, education, self_employed, loan_amount, loan_amount_term, credit_history, property_area, total_income]])

        # Start time for prediction
        start_time = time.time()

        # Model prediction
        prediction = model.predict(input_features)[0]

        # End time for prediction
        end_time = time.time()
        execution_time = round(end_time - start_time, 4)

        # Loan status
        result = 'Approved' if prediction == 1 else 'Rejected'
        
        # When generating the report
        report_generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return render_template('result.html', name=name, address=address, gender=gender, married=married, dependents=dependents, 
                               education=education, self_employed=self_employed, loan_amount=loan_amount, loan_amount_term=loan_amount_term, 
                               credit_history=credit_history, property_area=property_area, total_income=total_income, result=result, execution_time=execution_time, report_generation_time=report_generation_time)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=False)
