from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.datasets import load_iris

app = Flask(__name__)
iris = load_iris()
with open('model.pkl', 'rb') as model_file:     # rb means read and buffer
    model = pickle.load(model_file)
    
    
    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods =['POST'])
def predict():
    sepel_length = float(request.form['sl'])
    sepel_width = float(request.form['sw'])
    petal_length = float(request.form['pl'])
    petal_width = float(request.form['pw'])
    #inp = [[sepel_length, sepel_width, petal_length, petal_width]]
    #msg = model.predict(x)
    
    features = np.array([[ sepel_length, sepel_width, petal_length, petal_width]])
    prediction = model.predict(features)
    
    species = iris.target_names[prediction[0]]
    return render_template('home.html', pred_result = species)


if __name__ == '__main__':
    app.run(debug = True)
    