from flask import Flask, request, render_template
import os
import cv2
from predict import modcrop, shave, predict
from prepare_images import psnr, mse, compare_images, prepare_images
from PIL import Image
from zipfile import ZipFile

app = Flask(__name__)

UPLOAD_FOLDER = './static/input/'
OUTPUT_FOLDER = './static/output/'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route("/index", methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        image_file = request.files.get("image")
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            print("Image Location:", image_location)  # Debug print
            image_file.save(image_location)
            img = prepare_images(image_location, 2)
            ref, degraded, output, scores = predict(img, image_file.filename)
            output_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            output_image = Image.fromarray(output_image)
            output_location = os.path.join(OUTPUT_FOLDER, image_file.filename)

            output_image.save(output_location)
            return render_template('index.html', image_name=image_file.filename,
                                   psnr=scores[1][0],
                                   mse=scores[1][1])
    return render_template('index.html', image_name=None, psnr=0, mse=0)


@app.route('/chart')
def chart():
    return render_template('chart.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
