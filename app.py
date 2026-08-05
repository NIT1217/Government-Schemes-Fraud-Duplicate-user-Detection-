from flask import Flask
from flask import render_template
from flask import request
from flask import redirect,url_for
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "GET":
        return render_template("login.html")

    userid = request.form["value"]
    password = request.form["pass"]

    return redirect(url_for("upload"))
    
@app.route("/upload",methods=["GET","POST"])
def upload():
    if request.method=="GET":
        return render_template("upload.html")
    
    elif request.method=="POST":
        ID = request.files["IDProof"]
        image = request.files["selfie"]
              
    return  redirect(url_for("predict"))

@app.route("/predict", methods=["GET"])
def predict():
    return render_template("predict.html")
    
if __name__ == "__main__":
    app.run(debug=True)