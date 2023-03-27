

from flask import Flask, render_template, request
import pickle

model = pickle.load(open("logistic_regression_model.sav", "rb"))
target_names = list(
    map(lambda x: x.title(), ['setosa', 'versicolor', 'virginica']))

app = Flask(__name__)


@app.route("/")
def home():

    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(element) for element in request.form.values()]
    prediction = int(model.predict([features]))
    return render_template("home.html", prediction_text=target_names[prediction])


if __name__ == "__main__":
    app.run(debug=True, port=5533)
