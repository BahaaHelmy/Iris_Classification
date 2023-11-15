import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from statistics import mode
model_pkl_file1 = "model1.pkl"

with open(model_pkl_file1, 'rb') as file:
    model1 = pickle.load(file)
model_pkl_file2 = "model2.pkl"

with open(model_pkl_file2, 'rb') as file:
    model2 = pickle.load(file)
model_pkl_file3 = "model3.pkl"

with open(model_pkl_file3, 'rb') as file:
    model3 = pickle.load(file)
model_pkl_file4 = "model4.pkl"

with open(model_pkl_file4, 'rb') as file:
    model4 = pickle.load(file)
model_pkl_file5 = "model5.pkl"

with open(model_pkl_file5, 'rb') as file:
    model5 = pickle.load(file)
model_pkl_file6 = "model6.pkl"

with open(model_pkl_file6, 'rb') as file:
    model6 = pickle.load(file)
model_pkl_file7 = "model7.pkl"

with open(model_pkl_file7, 'rb') as file:
    model7 = pickle.load(file)
# Create flask app
app = Flask(__name__)

classes=['Not Disease','Disease']

@app.route("/")
def Home():
    return render_template("HB1.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features)
    features = features.reshape(-1, 1)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = features.reshape(1, -1)
    pred1=model1.predict(features)
    pred2=model2.predict(features)
    pred3=model3.predict(features)
    pred4=model4.predict(features)
    pred5=model5.predict(features)
    pred6=model6.predict(features)
    pred7=model7.predict(features)
    preds=[pred1[0],pred2[0],pred3[0],pred4[0],pred5[0],pred6[0],pred7[0]]
    majority_class = mode(preds)
    return render_template("HB.html", prediction_text = classes[majority_class])

if __name__ == "__main__":
    app.run(debug=True)
