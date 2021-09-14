from flask import Flask, request, render_template
import pickle

# initialize the Flask app
app = Flask(__name__)
# read the XGBoost model in read mode
model = pickle.load(open("models/model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    prediction = model.predict([int_features])
    return render_template(
        "home.html",
        prediction_text="Home Prediction is ${} USD".format(round(prediction[0], 2)),
    )


if __name__ == "__main__":
    app.run(debug=True)
