import numpy as np
from flask import Flask, request, render_template, url_for
# import pickle
import joblib
import warnings    
warnings.filterwarnings('ignore')

app = Flask(__name__)

filename = 'finalized_model.sav'
model = joblib.load(filename)
# model = pickle.load(open(filename, 'rb'))       

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]
    except ValueError:  
        return render_template('index.html', Prediction_text=f" Please enter Valid CGPA ")

    value = np.array(input_features)
    if value[0] < 290 or value[0] > 350:
            return render_template('index.html', Prediction_text=f" Please enter valid GRE score ")

    if value[1] < 90 or value[1] > 120:
            return render_template('index.html', Prediction_text=f" Please enter valid TOFEL score ")

    if value[5] < 6.0 or value[5] > 9.9:
            return render_template('index.html', Prediction_text=f" Please enter Valid CGPA ")   

    output = model.predict([value])[0] 
    return render_template('index.html', Prediction_text=f"The probability of addvision is  {np.round(output*100)}%.")

if __name__ == "__main__":
    app.run(debug=True)
