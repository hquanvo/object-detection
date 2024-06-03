from flask import Flask, render_template, request
from coco_object_detection import evaluate
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

@app.route("/")
def start():
    return render_template("index.html")

@app.route("/submit", methods = ["GET", "POST"])
def submit():
    if (request.method == "POST"):
        image = request.files["file"]
        print(type(image))
        #evaluate([image])
        return render_template("index.html")