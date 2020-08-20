# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions


## Loading the pretrained model
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('model.h5')


app = Flask(__name__)



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('first.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']


        model = load_model('model.h5')
        test_image = image.load_img(f, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        preds = model.predict(test_image)
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])
        return render_template('second.html',result=result)
    
if __name__ == '__main__':
  app.run(debug=True, use_reloader=False)




