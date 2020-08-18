import flask
from flask import Flask,render_template,url_for,request
import pickle
import base64
import numpy as np
import cv2
import tensorflow as tf

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21

#Our dictionary
label_dict = {0:'bird', 1:'cat', 2:'angel', 3:'bicycle', 4:'octopus', 5:'spider', 6:'flower', 7:'bee', 8:'mosquito', 9:'owl', 10:'yoga'}


#Initializing the Default Graph (prevent errors)
# graph = tf.compat.v1.get_default_graph()

# Use pickle to load in the pre-trained model.
# with open(f'model_cnn.pkl', 'rb') as f:
#         model = pickle.load(f)
model = tf.keras.models.load_model('model_cnn_final.h5')

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')

#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
        # global graph
        # with graph.as_default():
        if request.method == 'POST':
                final_pred = None
                #Preprocess the image : set the image to 28x28 shape
                #Access the image
                draw = request.form['url']
                #Removing the useless part of the url.
                draw = draw[init_Base64:]
                #Decoding into an image
                draw_decoded = base64.b64decode(draw)
                image = np.asarray(bytearray(draw_decoded), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                #Resizing and reshaping to keep the ratio.
                resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
                vect = np.asarray(resized, dtype="uint8")
                vect = vect.reshape(1, 28, 28, 1).astype('float32')
                print(vect.shape)
                #Predict
                my_prediction = model.predict(vect)
                #Getting the index of the maximum prediction
                index = np.argmax(my_prediction[0])
                #Associating the index and its value within the dictionnary
                final_pred = label_dict[index]

        return render_template('results.html', prediction =final_pred)

@app.route('/draw_partial')
def draw_partial():
        return render_template('multi_predict.html')

@app.route('/draw_complete')
def draw_complete():
        return render_template('multi_vae.html')


if __name__ == '__main__':
	app.run(debug=True)