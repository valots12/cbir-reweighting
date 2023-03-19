# Import libraries

from time import time
from PIL import Image
import cv2
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from skimage import io 
from keras import Model
from keras import Input
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from google.colab import drive
drive.mount('/content/gdrive')

# Data loading
#Importing the training dataset and convert to the desidered input dimension of the network

train_labels = pd.read_csv('train.csv')
train_labels['Classes'] = train_labels['Classes'].replace('F-16A/B', 'F-16AB')
train_labels['Classes'] = train_labels['Classes'].replace('F/A-18', 'FA-18')

similarity_df = pd.DataFrame(train_labels, columns=['filename','Classes','Labels'])
similarity_df['Similarity'] = np.nan

test_labels = pd.read_csv('test.csv')
test_labels['Classes'] = test_labels['Classes'].replace('F-16A/B', 'F-16AB')
test_labels['Classes'] = test_labels['Classes'].replace('F/A-18', 'FA-18')

# Definition network and feature extraction train
temp = keras.models.load_model('Model/densenet201.h5')

layer_name = 'dense_1'
newmodel = Model(inputs=temp.input, outputs=temp.get_layer(layer_name).output)
newmodel.summary()

train_newnet = np.load('Feature extraction/train_newnet_complete_dense.npy')

mu_array = np.load('Feature extraction/mu_vector.npy')
mu_array.shape

std_array = np.load('Feature extraction/std_vector.npy')
std_array.shape

def normalizeFeatureMatrix(feat_matrix):

  for i in range(0,feat_matrix.shape[1]):

    # 1.normalizing
    feat_matrix[:,i] = (feat_matrix[:,i] - mu_array[i]) / (3 * std_array[i]) 

    # 2.force outliers < -1 to -1 and > 1 to 1
    feat_matrix[:,i] = np.clip(feat_matrix[:,i], -1, 1)

    # 3.[0,1] range
    feat_matrix[:,i] = (feat_matrix[:,i] + 1) / 2

  return feat_matrix

train_newnet = normalizeFeatureMatrix(train_newnet)

# Feature extraction query
# Use the estimated network as feature extractor for each image

def mobilenet_features(img):
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    f = newmodel.predict(x, verbose=False)
    return f

"""
Normalizing features using the following formula:
f' = (f - mu) / 3 * sigma

Forcing the features in the [0,1] range
f' = (f + 1) / 2
"""

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
  img_path = 'images/'+img_name
  image = kimage.load_img(img_path, target_size=(224, 224))
  features = mobilenet_features(image)

  features = np.array(features)
  return normalizeFeatureVector(features)

"""
Evaluation results with test query

Evaluate distance between each image and the query one using Minkowski:
D(I,Q) = sum (wi * |fi(image) - fi(query)|)
"""

def getMinkowskiSimilarity(A, B, weights=None):
  if weights is None:
    return sum(abs(a-b) for a,b in zip(A, B))
  else:
    return sum(w*abs(a-b) for a,b,w in zip(A, B, weights))

# Find best N results more similar to the query image
def getQueryLabel(img_name, train=False):
  if train:
    df_labels = train_labels
  else:
    df_labels = test_labels

  df_labels = df_labels.reset_index()
  return df_labels[df_labels['filename']==img_name]['Classes'].values[0]

# Plot similar N images with given image and similar images dataframe
def plotSimilarImages(img_file, similar_df, number_retrieved):
  img = kimage.load_img('images/' + img_file, target_size=(224, 224))

  img_class = getQueryLabel(img_file)
  fig, axarr = plt.subplots(2,3,figsize=(15, 8))
  axarr[0,0].imshow(img)
  axarr[0,0].set_title("TEST IMAGE - " + "\nClass: " + img_class)
  axarr[0,0].axis('off')

  j, k, m = 0, 0, 1
  for index, sim in similar_df.iterrows():
    sim_class = getQueryLabel(sim['filename'], train=True)
    similarity = sim['Similarity']

    similar = kimage.load_img('images/' + sim['filename'], target_size=(224, 224))
    axarr[k,m].imshow(similar)
    axarr[k,m].set_title("Similarity: %.2f" % similarity + "\nClass: " + sim_class)
    axarr[k,m].axis('off')

    m += 1
    if m == 3 and k != 1:
      k += 1
      m = 0

    j += 1
    if j == 5:
      break

  plt.tight_layout()
  plt.show()

def evaluate(sim_df, query_class, number_retrieved):
  accuracy = 0
  for i in sim_df['Classes'][0:number_retrieved].values:
    if i == query_class:
      accuracy += 1
  return accuracy/number_retrieved*100

# Get and plot similar images for given image path and features dataframe
def getSimilarImages(img_file, number_retrieval, evaluation=False):
  img_features = getFeatureVector(img_file)
  img_features = img_features[0, :]

  for i in range(0,similarity_df.shape[0]):
    similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:])

  sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)

  if evaluation:
    return evaluate(sorted_df, getQueryLabel(img_file), number_retrieval)

  plotSimilarImages(img_file, sorted_df.head(number_retrieval), number_retrieval)

# Get similar images of test image
t0 = time()
getSimilarImages(test_labels['filename'][0], 10)
print("Time required: %0.2f seconds." % (time() - t0))


"""
Rebalancing - strategy 1

Update weights type 1
w' = (e + sigma(retrieved images)) / (e + sigma(relevant images))
e = 0.0001
"""

def getWeightsRF_type1(query_features, retrieval_features, query_name, eps=0.0001):

  retrieval_features.loc[:,('index')] = retrieval_features.index

  # recognise relevant and retrieved documents
  relevant_imgs = []
  retrieval_imgs = []
  for i in range(0,retrieval_features.shape[0]):
    if retrieval_features['Classes'].to_list()[i] == query_name:
      relevant_imgs.append(retrieval_features['index'].to_list()[i])
    retrieval_imgs.append(retrieval_features['index'].to_list()[i])

  if len(relevant_imgs)>3:

    relevant_features = train_newnet[relevant_imgs]
    retrieval_features = train_newnet[retrieval_imgs]

    new_weights = []
    for i in range(0,query_features.shape[0]):
      new_weights.append((eps + np.std(retrieval_features[:,i])) / (eps + np.std(relevant_features[:,i])))

    return new_weights

  else:
    return [1 for i in range(0,retrieval_features.shape[0])]


# Get and plot similar images for given image path and features dataframe
def getSimilarImages_RF_type1(img_file, num_rounds, number_retrieval, evaluation=False):

  df_output = pd.DataFrame([img_file], columns=['filename'])

  img_features = getFeatureVector(img_file)
  img_features = img_features[0, :]

  for i in range(0,similarity_df.shape[0]):
    similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:])

  sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)

  if evaluation:
    df_output['Evaluation_0'] = evaluate(sorted_df, getQueryLabel(img_file), number_retrieval)
  else:
    plotSimilarImages(img_file, sorted_df.head(number_retrieval), number_retrieval)

  j = 1
  while j <= num_rounds:

    if j == 1:
      tmp_weights = getWeightsRF_type1(img_features, sorted_df.head(number_retrieval), getQueryLabel(img_file))
      old_weights = [1 for i in range(0,len(tmp_weights))]
      new_weights = [0.9*x[0] + 0.1*x[1] for x in zip(old_weights, tmp_weights)]
    else:
      tmp_weights = getWeightsRF_type1(img_features, sorted_df.head(number_retrieval), getQueryLabel(img_file))
      old_weights = new_weights
      new_weights = [0.9*x[0] + 0.1*x[1] for x in zip(old_weights, tmp_weights)]

    for i in range(0,similarity_df.shape[0]):
      similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:], new_weights)

    sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)

    if evaluation:
      df_output['Evaluation_' + str(j)] = evaluate(sorted_df, getQueryLabel(img_file), number_retrieval)
    else:
      print('New iteration')
      plotSimilarImages(img_file, sorted_df.head(number_retrieval), number_retrieval)

    j += 1
  
  return df_output


"""
Rebalancing - strategy 2

Update weights type 2

sigma = 1 - (no. of non-relevant images located inside the dominant range of relevant samples) /
(total no. non-relevant images among the retrieved images)

PS The dominant range of a feature component is found by the minimum and maximum values from the
set of relevant samples.

w' = sigma / (e + sigma(relevant images))
"""

def getWeightsRF_type2(query_features, retrieval_features, query_name, eps=0.0001):

  retrieval_features.loc[:,('index')] = retrieval_features.index

  # recognise relevant and retrieved documents
  relevant_imgs = []
  norelevant_imgs = []
  retrieval_imgs = []

  for i in range(0,retrieval_features.shape[0]):
    if retrieval_features['Classes'].to_list()[i] == query_name:
      relevant_imgs.append(retrieval_features['index'].to_list()[i])
    else:
      norelevant_imgs.append(retrieval_features['index'].to_list()[i])
    retrieval_imgs.append(retrieval_features['index'].to_list()[i])

  if len(relevant_imgs)>3 and len(norelevant_imgs)>3:

    relevant_features = train_newnet[relevant_imgs]
    norelevant_features = train_newnet[norelevant_imgs]
    retrieval_features = train_newnet[retrieval_imgs]
    
    new_weights = []
    for i in range(0,query_features.shape[0]):

      min_rel = np.argmin(relevant_features[:,i])
      max_rel = np.argmax(relevant_features[:,i])

      phi = 0
      for norel in range(0,norelevant_features.shape[0]):
        if norelevant_features[norel, i] >= min_rel and norelevant_features[norel, i] <= max_rel:
          phi += 1
      
      F = norelevant_features.shape[0]
      delta = 1 - (phi/F)

      new_weights.append(delta / (eps + np.std(relevant_features[:,i])))

    return new_weights

  else:
    return [1 for i in range(0,retrieval_features.shape[0])]

# Get and plot similar images for given image path and features dataframe
def getSimilarImages_RF_type2(img_file, num_rounds, number_retrieval, evaluation=False):

  df_output = pd.DataFrame([img_file], columns=['filename'])

  img_features = getFeatureVector(img_file)
  img_features = img_features[0, :]

  for i in range(0,similarity_df.shape[0]):
    similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:])

  sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)

  if evaluation:
    df_output['Evaluation_0'] = evaluate(sorted_df, getQueryLabel(img_file), number_retrieval)
  else:
    plotSimilarImages(img_file, sorted_df.head(number_retrieval), number_retrieval)

  j = 1
  while j <= num_rounds:

    if j == 1:
      tmp_weights = getWeightsRF_type2(img_features, sorted_df.head(number_retrieval), getQueryLabel(img_file))
      old_weights = [1 for i in range(0,len(tmp_weights))]
      new_weights = [0.9*x[0] + 0.1*x[1] for x in zip(old_weights, tmp_weights)]
    else:
      tmp_weights = getWeightsRF_type2(img_features, sorted_df.head(number_retrieval), getQueryLabel(img_file))
      old_weights = new_weights
      new_weights = [0.9*x[0] + 0.1*x[1] for x in zip(old_weights, tmp_weights)]

    for i in range(0,similarity_df.shape[0]):
      similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:], new_weights)

    sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)

    if evaluation:
      df_output['Evaluation_' + str(j)] = evaluate(sorted_df, getQueryLabel(img_file), number_retrieval)
    else:
      print('New iteration')
      plotSimilarImages(img_file, sorted_df.head(number_retrieval), number_retrieval)

    j += 1
  
  return df_output


"""
Rebalancing - strategy 3

Update weights type 3
w' = sigma * (e + sigma(retrieved images)) / (e + sigma(relevant images))
"""

def getWeightsRF_type3(query_features, retrieval_features, query_name, eps=0.0001):

  retrieval_features.loc[:,('index')] = retrieval_features.index

  # recognise relevant and retrieved documents
  relevant_imgs = []
  norelevant_imgs = []
  retrieval_imgs = []

  for i in range(0,retrieval_features.shape[0]):
    if retrieval_features['Classes'].to_list()[i] == query_name:
      relevant_imgs.append(retrieval_features['index'].to_list()[i])
    else:
      norelevant_imgs.append(retrieval_features['index'].to_list()[i])
    retrieval_imgs.append(retrieval_features['index'].to_list()[i])

  if len(relevant_imgs)>3 and len(norelevant_imgs)>3:

    relevant_features = train_newnet[relevant_imgs]
    norelevant_features = train_newnet[norelevant_imgs]
    retrieval_features = train_newnet[retrieval_imgs]
    
    new_weights = []
    for i in range(0,query_features.shape[0]):

      min_rel = np.argmin(relevant_features[:,i])
      max_rel = np.argmax(relevant_features[:,i])

      phi = 0
      for norel in range(0,norelevant_features.shape[0]):
        if norelevant_features[norel, i] >= min_rel and norelevant_features[norel, i] <= max_rel:
          phi += 1
      
      F = norelevant_features.shape[0]
      delta = 1- (phi/F)

      new_weights.append(delta * (eps + np.std(retrieval_features[:,i])) / (eps + np.std(relevant_features[:,i])))

    return new_weights

  else:
    return [1 for i in range(0,retrieval_features.shape[0])]

def getWeightsRF_type1(query_features, retrieval_features, query_name, eps=0.0001):

  retrieval_features.loc[:,('index')] = retrieval_features.index

  # recognise relevant and retrieved documents
  relevant_imgs = []
  retrieval_imgs = []
  for i in range(0,retrieval_features.shape[0]):
    if retrieval_features['Classes'].to_list()[i] == query_name:
      relevant_imgs.append(retrieval_features['index'].to_list()[i])
    retrieval_imgs.append(retrieval_features['index'].to_list()[i])

  if len(relevant_imgs)>3:

    relevant_features = train_newnet[relevant_imgs]
    retrieval_features = train_newnet[retrieval_imgs]

    new_weights = []
    for i in range(0,query_features.shape[0]):
      new_weights.append((eps + np.std(retrieval_features[:,i])) / (eps + np.std(relevant_features[:,i])))

    return new_weights

  else:
    return [1 for i in range(0,retrieval_features.shape[0])]

# Get and plot similar images for given image path and features dataframe
def getSimilarImages_RF_type3(img_file, num_rounds, number_retrieval, evaluation=False):

  df_output = pd.DataFrame([img_file], columns=['filename'])

  img_features = getFeatureVector(img_file)
  img_features = img_features[0, :]

  for i in range(0,similarity_df.shape[0]):
    similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:])

  sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)

  if evaluation:
    df_output['Evaluation_0'] = evaluate(sorted_df, getQueryLabel(img_file), number_retrieval)
  else:
    plotSimilarImages(img_file, sorted_df.head(number_retrieval), number_retrieval)

  j = 1
  while j <= num_rounds:

    if j == 1:
      tmp_weights = getWeightsRF_type3(img_features, sorted_df.head(number_retrieval), getQueryLabel(img_file))
      old_weights = [1 for i in range(0,len(tmp_weights))]
      new_weights = [0.9*x[0] + 0.1*x[1] for x in zip(old_weights, tmp_weights)]
    else:
      tmp_weights = getWeightsRF_type3(img_features, sorted_df.head(number_retrieval), getQueryLabel(img_file))
      old_weights = new_weights
      new_weights = [0.9*x[0] + 0.1*x[1] for x in zip(old_weights, tmp_weights)]

    for i in range(0,similarity_df.shape[0]):
      similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:], new_weights)

    sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)

    if evaluation:
      df_output['Evaluation_' + str(j)] = evaluate(sorted_df, getQueryLabel(img_file), number_retrieval)
    else:
      print('New iteration')
      plotSimilarImages(img_file, sorted_df.head(number_retrieval), number_retrieval)

    j += 1
  
  return df_output

def getSimilarImages_RF_type1(img_file, num_rounds, number_retrieval, evaluation=False):

  df_output = pd.DataFrame([img_file], columns=['filename'])

  img_features = getFeatureVector(img_file)
  img_features = img_features[0, :]

  for i in range(0,similarity_df.shape[0]):
    similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:])

  sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)

  if evaluation:
    df_output['Evaluation_0'] = evaluate(sorted_df, getQueryLabel(img_file), number_retrieval)
  else:
    plotSimilarImages(img_file, sorted_df.head(number_retrieval), number_retrieval)

  j = 1
  while j <= num_rounds:

    if j == 1:
      tmp_weights = getWeightsRF_type1(img_features, sorted_df.head(number_retrieval), getQueryLabel(img_file))
      old_weights = [1 for i in range(0,len(tmp_weights))]
      new_weights = [0.9*x[0] + 0.1*x[1] for x in zip(old_weights, tmp_weights)]
    else:
      tmp_weights = getWeightsRF_type1(img_features, sorted_df.head(number_retrieval), getQueryLabel(img_file))
      old_weights = new_weights
      new_weights = [0.9*x[0] + 0.1*x[1] for x in zip(old_weights, tmp_weights)]

    for i in range(0,similarity_df.shape[0]):
      similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:], new_weights)

    sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)

    if evaluation:
      df_output['Evaluation_' + str(j)] = evaluate(sorted_df, getQueryLabel(img_file), number_retrieval)
    else:
      print('New iteration')
      plotSimilarImages(img_file, sorted_df.head(number_retrieval), number_retrieval)

    j += 1