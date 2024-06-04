import cv2
from flask import Flask, render_template, request, redirect
from coco_object_detection import evaluate, show_image
from werkzeug.utils import secure_filename
import os
from pathlib import Path

app = Flask(__name__, static_folder="./static")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = "./images"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def start():
    Path("./static").mkdir(parents=True, exist_ok=True)
    Path("./images").mkdir(parents=True, exist_ok=True)

    return render_template("index.html")

@app.route("/submit", methods = ["GET", "POST"])
def submit():
    if (request.method == "POST"):

        file = request.files["file"]

        if (file and allowed_file(file.filename)):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            img = cv2.imread(path)

            annotated_imgs = evaluate([img])

            cv2.imwrite(os.path.join("./static", filename), annotated_imgs[0])

        return render_template("index.html", image = os.path.join("./static", filename))