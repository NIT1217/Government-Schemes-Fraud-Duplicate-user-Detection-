import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from model.predict import prediction

app = Flask(__name__)

UPLOAD_FOLDER = "upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "GET":
        return render_template("index.html")

    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "GET":
        return render_template("login.html")

    userid = request.form["value"]
    password = request.form["pass"]

    # (Optional) Validate userid/password here

    return redirect(url_for("upload"))


@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "GET":
        return render_template("upload.html")

    # Check files
    if "IDProof" not in request.files or "selfie" not in request.files:
        return "Please upload both the ID Proof and Selfie."

    ID = request.files["IDProof"]
    image = request.files["selfie"]

    if ID.filename == "" or image.filename == "":
        return "No file selected."

    # Secure filenames
    id_filename = secure_filename(ID.filename)
    selfie_filename = secure_filename(image.filename)

    id_path = os.path.join(app.config["UPLOAD_FOLDER"], id_filename)
    selfie_path = os.path.join(app.config["UPLOAD_FOLDER"], selfie_filename)

    # Save files
    ID.save(id_path)
    image.save(selfie_path)

    # Run AI model
    report = prediction(selfie_path, id_path)

    # Display result
    return render_template("predict.html", report=report)


if __name__ == "__main__":
    app.run(debug=True)