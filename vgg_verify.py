from matplotlib import pyplot
from PIL import Image
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input
import time
import os
import numpy as np
import re

THRESHOLD = 0.28
 
# extract a single face from a given photograph
def extract_face(detector, filename, index, img = None, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename) if img is None else img

    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]  

    # resize pixels to the model size
    try:
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)  
        return index, face_array
    except:
        print("FAIL: ", filename)
        return index, None
        
# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(detector, model, filenames):
    faces = []
    labels = []
    # extract faces
    for idx, f in enumerate(filenames):
        index, face = extract_face(detector, f, idx)
        if face is not None:
            faces.append(face)
            labels.append(f)
            
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')

    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    
    # perform prediction
    yhat = model.predict(samples)

    return labels, yhat

def get_embedding_img(detector, model, img, index = 0):
    _, face = extract_face(detector, None, index, img)
    samples = np.asarray([face], 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    
    # perform prediction
    yhat = model.predict(samples)

    return yhat
 
# determine if a candidate face is a match for a known face
def is_match(detector, model, known_embeddings, candidate_img, thresh=THRESHOLD):
    # calculate distance between embeddings
    minscore = thresh
    minindex = -1
    candidate_embedding = get_embedding_img(detector, model, candidate_img)
    for index in range(len(known_embeddings)):
        score = cosine(known_embeddings[index], candidate_embedding[0])
        if score < minscore:
            minscore = score
            minindex = index

    #return score
    if minscore < thresh:
        print('face is MATCH (%.3f <= %.3f)' % (minscore, thresh))
    else:
        print('face is NOT MATCH (%.3f > %.3f)' % (minscore, thresh))
        minindex = -1
        
    return minindex, minscore
 
def setup_emb(detector, model, path):
    filenames = [f for f in os.listdir(path) if re.match(r'.*?\.jpg', f)]
    print(filenames)
    labels, embeddings = get_embeddings(detector, model, filenames)
    
    print("="*10, len(embeddings))
    print("="*10, len(labels))
    
    np.save('embeddings.npy', embeddings)
    np.save('labels.npy', labels)
 
def load_emb(lblname, embname):
    return np.load(lblname), np.load(embname)

