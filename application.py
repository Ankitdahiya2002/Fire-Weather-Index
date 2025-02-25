import pickle
import numpy as np
from flask import Flask, request, jsonify,render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application
with open('models/ridge.pkl', 'rb') as file1:  # load pickle file
    ridge_model= pickle.load(file1)             
 
with open ('models/scaler.pkl','rb')as file2:    # load pickle file
    standard_scaler = pickle.load(file2)

@app.route('/')                          # index html file
def index():
    return render_template("index.html")



# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = eval(request.form.get('Temperature'))
        RH = eval(request.form.get('RH'))
        Ws = eval(request.form.get('Ws'))
        Rain = eval(request.form.get('Rain'))
        FFMC = eval(request.form.get('FFMC'))
        DMC = eval(request.form.get('DMC'))
        ISI = eval(request.form.get('ISI'))
        Classes = eval(request.form.get('Classes'))
        Region = eval(request.form.get('Region'))
        
    
    
    
        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]) 
        result = ridge_model.predict(new_data_scaled)
    
        return render_template('home.html',result = result[0])  
    
    

    else:
        return render_template('home.html')  
        
            



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

    