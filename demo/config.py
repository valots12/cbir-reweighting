import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras import Input
from tensorflow.keras.preprocessing import image as kimage


# Read features and vector of averages and standard deviations
train_newnet = np.load('./static/feature/train_newnet.npy')
mu_array = np.load('./static/feature/mu_vector.npy')
std_array = np.load('./static/feature/std_vector.npy')


temp = keras.models.load_model('./static/feature/densenet201_final_task2.h5')
layer_name = 'dense_1'
newmodel = Model(inputs=temp.input, outputs=temp.get_layer(layer_name).output)


def normalizeFeatureMatrix(feat_matrix):
  for i in range(0,feat_matrix.shape[1]):
    # 1.normalizing
    feat_matrix[:,i] = (feat_matrix[:,i] - mu_array[i]) / (3 * std_array[i]) 
    # 2.force outliers < -1 to -1 and > 1 to 1
    feat_matrix[:,i] = np.clip(feat_matrix[:,i], -1, 1)
    # 3.[0,1] range
    feat_matrix[:,i] = (feat_matrix[:,i] + 1) / 2
  return feat_matrix


def mobilenet_features(img):
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    f = newmodel.predict(x, verbose=False)
    return f


def normalizeFeatureVector(feat_array):
  for i in range(0, len(feat_array)):
    # 1.normalizing
    feat_array[:,i] = (feat_array[:,i] - mu_array[i]) / (3 * std_array[i]) 
    # 2.force outliers < -1 to -1 and > 1 to 1
    feat_array[:,i] = np.clip(feat_array[:,i], -1, 1)
    # 3.[0,1] range
    feat_array[:,i] = (feat_array[:,i] + 1) / 2
  return feat_array


def getFeatureVector(img_name):
  # Load file and extract features
  img_path = 'static/uploaded/'+img_name
  image = kimage.load_img(img_path, target_size=(224, 224))
  features = mobilenet_features(image)

  features = np.array(features)
  return normalizeFeatureVector(features)


def getMinkowskiSimilarity(A, B, weights=None):
  if weights is None:
    return sum(abs(a-b) for a,b in zip(A, B))
  else:
    return sum(w*abs(a-b) for a,b,w in zip(A, B, weights))


def getQueryLabel(img_name):
  train_labels = train_labels.reset_index()
  return train_labels[train_labels['filename']==img_name]['Classes'].values[0]


def getWeightsRF_type1(query_features, relevant_features, retrieval_features, eps=0.0001):

  new_weights = []
  for i in range(0,query_features.shape[0]):
    new_weights.append((eps + np.std(retrieval_features[:,i])) / (eps + np.std(relevant_features[:,i])))
  return new_weights