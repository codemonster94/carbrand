import os
from flask import Flask
from flask import request
from flask import render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('model_resnet50.h5')

UPLOAD_FOLDER="uploads"




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
@app.route('/result', methods=['GET'])
def result():
    # Main page
    return render_template('result.html')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The Car iS Audi"
    elif preds == 1:
        preds = "The Car is ferrari"
    elif preds == 2:
        preds = "The Car Is lamborghini"
    elif preds == 3:
        preds = "The Car Is mercedes"
    else:
        preds = " no idea what it is! i am stupid"

    return render_template('result.html', result=preds)



@app.route("/", methods=['GET','POST'])
def upload_predict():
    if request.method=='POST':

        f=request.files['file']

        image_location= os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(image_location)
        preds = model_predict(image_location, model)
        result=preds
        return result
    return None




if __name__== "__main__":
    app.run(port=8080, debug=True)
