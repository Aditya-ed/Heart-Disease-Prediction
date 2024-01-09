import numpy as np
from flask import Flask,render_template,url_for,request
import pickle

app = Flask(__name__)
model = pickle.load(open('RF.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/menu')
def menu():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = []
    for x in request.form.values():
        int_features.append(x)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    if (output == 0):
        text="you don't have any symptoms of a Heart Disease."
    else :
        text="you have the symptoms of having Heart Disease. Please consult a Doctor."
    #print(int_features)
    return render_template('index.html',prediction_text='The model predicted  {}'.format(text))


if __name__ == "__main__":
    app.run(debug=True)
