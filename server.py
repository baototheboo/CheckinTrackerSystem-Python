import sys
from flask import Flask, request, jsonify, abort
from flask.wrappers import Response
from flask_cors import CORS, cross_origin
import base64
import numpy as np
import cv2
import os
import shutil
import io
import re
import vgg_verify
from scipy import misc
from datetime import datetime
from pathlib import Path
from keras_vggface.vggface import VGGFace
from keras import backend as K
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import matplotlib.image as mpimg
from keras.models import load_model
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)
CORS(app, support_credentials=True)

EMBEDDINGS_KEY = 'embeddings.npy'
LABELS_KEY = 'labels.npy'
EMBEDDINGS_KEY_BAK = 'embeddings.npy.bak'
LABELS_KEY_BAK = 'labels.npy.ank'

global detector
global model
global graph
global emotion_model
# create the detector, using default weights
graph = tf.compat.v1.get_default_graph()
sess =  tf.compat.v1.Session()

set_session(sess)

# create the detector, using default weights
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

emotion_model = load_model("models/model_v6_23.hdf5")

labels, embeddings = vgg_verify.load_emb(LABELS_KEY, EMBEDDINGS_KEY)
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

print("Length labels and embeddedings")
print(len(labels))
print(len(embeddings))

@app.route('/verify-staff', methods=['POST'])
def verify_staff_keras_vgg():
    global sess
    global graph

    fdict = []
    data = request.json
    time_verify = data["timeVerify"]
    image = None
    verify_base_path = "images/verify"
    date_now= datetime.today().strftime('%Y_%m_%d')
    Path("%s/%s"%(verify_base_path,date_now)).mkdir(parents=True, exist_ok=True)
    for b64img in data["imgs"]:
        try:
            base64_data = re.sub('^data:image/.+;base64,', '', b64img)
            decoded_data = base64.b64decode(base64_data)
            byteimage = io.BytesIO(decoded_data)
            image = mpimg.imread(byteimage, format='JPG')
        except Exception as e:
            # cv2.imwrite('%s/%s/%s.png'%(verify_base_path,date_now,time_verify),image_decoded)
            print('Error verify: ', str(e))
            pass           

    with graph.as_default():
        set_session(sess)
        matchidx, score = vgg_verify.is_match(detector, model, embeddings, image)
        if matchidx is -1:
            return jsonify({ "partId": "", "probability": ""})
        label = labels[matchidx]
        print('='*50)
        print(matchidx, label, score)
        index = label.find('_')
        partId = label[0:index]

        image = cv2.resize(image, (48,48))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        predicted_class = np.argmax(emotion_model.predict(image))
        label_map = dict((v,k) for k,v in emotion_dict.items()) 
        predicted_label = label_map[predicted_class]

        print("predicted_label: ", predicted_label)

        print("Length labels and embeddedings")
        print(len(labels))
        print(len(embeddings))

        return jsonify({ "partId": partId, "probability": "%f"%(1 - score), "emotion": predicted_label})           

@app.route('/setup-staff', methods=['POST'])
def setup_staff():
    global embeddings
    global labels

    data = request.json
    staff_id = data["staffId"]
    staff_name = data["staffName"]
    # time_verify = data["timeVerify"]
    # accuracy_image = data["accuracyImage"]
    prewhiten_arr = []
    llabel = np.array([])
    setup_base_path = "images/setup"
    Path("%s"%(setup_base_path)).mkdir(parents=True, exist_ok=True)

    for cnt, b64img in enumerate(data["imgs"]):
        try:
            base64_data = re.sub('^data:image/.+;base64,', '', b64img)
            decoded_data = base64.b64decode(base64_data)
            byteimage = io.BytesIO(decoded_data)
            image = mpimg.imread(byteimage, format='JPG')
            with graph.as_default():
                set_session(sess)
                emb = vgg_verify.get_embedding_img(detector, model, image)
                if emb is not None:
                    sectime = int(datetime.now().timestamp())
                    label_name = '%s_%s'%(staff_id, sectime)
                    labels = np.append(labels, label_name)
                    embeddings = np.vstack((embeddings, emb))
                
        except Exception as e:
            print('Error: \n', str(e))
            continue
    
    try:
        if (len(embeddings) != len(labels)):
            raise MissMatchException("Length of Setup data miss match")

        shutil.copy2(EMBEDDINGS_KEY, EMBEDDINGS_KEY_BAK)
        shutil.copy2(LABELS_KEY, LABELS_KEY_BAK)
        np.save(EMBEDDINGS_KEY, embeddings)
        np.save(LABELS_KEY, labels)
        after_labels, after_embeddings = vgg_verify.load_emb(LABELS_KEY, EMBEDDINGS_KEY)

        if (len(after_embeddings) != len(after_labels)):
            os.remove(EMBEDDINGS_KEY)
            os.remove(LABELS_KEY)
            os.rename(EMBEDDINGS_KEY_BAK, EMBEDDINGS_KEY)
            os.rename(LABELS_KEY_BAK, LABELS_KEY)
            raise MissMatchException("Length lables and embeddings after setup miss match")
        else:
            os.remove(EMBEDDINGS_KEY_BAK)
            os.remove(LABELS_KEY_BAK)

    except Exception as e:
        print('Error: \n', str(e))
        abort(500, str(e))

    print("Length labels and embeddedings")
    print(len(labels))
    print(len(embeddings))

    return "OK"

@app.route('/delete-staff', methods=['POST'])
def delete_staff():
    global embeddings
    global labels

    data = request.json
    staff_id = data["staffId"]

    #embedding_key = np.where(labels == staff_id)
    arrIndex = np.flatnonzero(np.core.defchararray.find(labels, staff_id) != -1)
    if len(arrIndex) == 0:
        error_message = {'error': 'No image setup'}
        return jsonify(error_message), 404
    else:
        arrIndex[::-1].sort()
        for index in arrIndex:
            if labels[index].startswith(staff_id):
                embeddings = np.delete(embeddings, index, 0)
                labels = np.delete(labels, index)
        np.save(EMBEDDINGS_KEY, embeddings)
        np.save(LABELS_KEY, labels)

        print("Length labels and embeddedings")
        print(len(labels))
        print(len(embeddings))
        return "OK"

@app.route('/delete-all-staff', methods=['POST'])
def delete_all():
    global embeddings
    global labels

    embeddings = np.delete(embeddings, np.s_[:], axis=0)

    labels = []

    np.save(EMBEDDINGS_KEY, embeddings)
    np.save(LABELS_KEY, labels)

    return "OK"

class MissMatchException(Exception):
    def __init__(self, value):
        self.value = value

if __name__ == '__main__':
   app.run(host="0.0.0.0", port="5001", threaded=True)
