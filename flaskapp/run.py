from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        ABV = request.form["ABV"]
        Min_IBU = request.form["Min_IBU"]
        Max_IBU = request.form["Max_IBU"]
        Astringency = request.form["Astringency"]
        Body = request.form["Body"]
        Alcohol = request.form["Alcohol"]
        Bitter = request.form["Bitter"]
        Sweet = request.form["Sweet"]
        Sour = request.form["Sour"]
        Salty = request.form["Salty"]
        Fruits = request.form["Fruits"]
        Hoppy = request.form["Hoppy"]
        Spices = request.form["Spices"]
        Malty = request.form["Malty"]
        
        X = np.array([[float(ABV), int(Min_IBU), int(Max_IBU), int(Astringency), int(Body), int(Alcohol), int(Bitter), int(Sweet),int(Sour),int(Salty),int(Fruits),int(Hoppy),int(Spices), int(Malty)]])
        
        pred = model.predict(X)[0][1]
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
