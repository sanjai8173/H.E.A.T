from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pickle", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Wind_Speed = float(request.form['Wind_Speed'])
        Pressure = float(request.form['Pressure'])

        data = [Temperature, Humidity, Wind_Speed, Pressure]
        prediction = model.predict([data])[0]
        return render_template("result.html", prediction=prediction)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
