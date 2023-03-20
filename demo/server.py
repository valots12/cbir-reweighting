import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from config import *

app = Flask(__name__)


# Read image features
train_labels = pd.read_csv('./static/feature/train.csv')
train_labels['Classes'] = train_labels['Classes'].replace('F-16A/B', 'F-16AB')
train_labels['Classes'] = train_labels['Classes'].replace('F/A-18', 'FA-18')

similarity_df = pd.DataFrame(train_labels, columns=['filename','Classes','Labels'])
similarity_df['Similarity'] = np.nan

train_newnet = np.load('./static/feature/train_newnet.npy')
train_newnet = normalizeFeatureMatrix(train_newnet)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    file = request.files['query_img']

    # Save query image
    img = Image.open(file.stream)  # PIL image
    uploaded_img_path = "static/uploaded/" + file.filename
    img.save(uploaded_img_path)

    # Run search
    img_features = getFeatureVector(file.filename)
    img_features = img_features[0, :]

    for i in range(0,similarity_df.shape[0]):
        similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:])

    sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)
    sorted_df['index_orig'] = sorted_df.index
    sorted_df = sorted_df.reset_index()

    img_paths = []
    img_paths_str = []
    img_dists = []
    img_indexes = []
    scores = []
    for i in range(0,6):
        tmp_path = Path("./static/img") / (sorted_df.loc[i,'filename'])
        tmp_dist = sorted_df.loc[i,'Similarity']
        tmp_index = sorted_df.loc[i,'index_orig']
        img_paths.append(tmp_path)
        img_paths_str.append(str(tmp_path))
        img_dists.append(tmp_dist)
        img_indexes.append(tmp_index)
        scores.append((sorted_df.loc[i,'Classes'], tmp_path))

    with open('paths_best.txt', 'w') as fp:
        for item in img_paths_str:
            fp.write("%s\n" % item)

    with open('indexes_best.txt', 'w') as fp:
        for item in img_indexes:
            fp.write("%s\n" % item)

    with open('uploaded_img.txt', 'w') as fp:
        fp.write("%s\n" % file.filename)

    with open('num_rounds.txt', 'w') as fp:
        fp.write("%s\n" % 0)

    return render_template('index.html',
                            query_path=uploaded_img_path,
                            scores=scores)


@app.route('/rebalance', methods=['POST'])
def rebalance():

    file = open('paths_best.txt','r')
    list_paths = file.read()
    list_paths = list_paths.split("\n")
    file.close()

    file = open('indexes_best.txt','r')
    list_indexes = file.read()
    list_indexes = list_indexes.split("\n")
    list_indexes = [int(x) for x in range(0,len(list_indexes))]
    file.close()

    file = open('uploaded_img.txt','r')
    uploaded_img = file.read()
    uploaded_img = uploaded_img.split("\n")[0]
    uploaded_img_path = "static/uploaded/" + uploaded_img
    file.close()

    file = open('num_rounds.txt','r')
    num_rounds = file.read()
    num_rounds = num_rounds.split("\n")[0]
    file.close()

    num_rounds = int(num_rounds)
    num_rounds+=1

    list_relevant_paths = []
    list_relevant_indexes = []

    importance1 = request.form.get("Importance1")
    if importance1 == 'Relevant':
        list_relevant_paths.append(list_paths[0])
        list_relevant_indexes.append(list_indexes[0])

    importance2 = request.form.get("Importance2")
    if importance2 == 'Relevant':
        list_relevant_paths.append(list_paths[1])
        list_relevant_indexes.append(list_indexes[1])

    importance3 = request.form.get("Importance3")
    if importance3 == 'Relevant':
        list_relevant_paths.append(list_paths[2])
        list_relevant_indexes.append(list_indexes[2])

    importance4 = request.form.get("Importance4")
    if importance4 == 'Relevant':
        list_relevant_paths.append(list_paths[3])
        list_relevant_indexes.append(list_indexes[3])
    
    importance5 = request.form.get("Importance5")
    if importance5 == 'Relevant':
        list_relevant_paths.append(list_paths[4])
        list_relevant_indexes.append(list_indexes[4])

    importance6 = request.form.get("Importance6")
    if importance6 == 'Relevant':
        list_relevant_paths.append(list_paths[5])
        list_relevant_indexes.append(list_indexes[5])

    # Run search
    img_features = getFeatureVector(uploaded_img)
    img_features = img_features[0, :]

    if len(list_relevant_indexes)>1:
        new_weights =  getWeightsRF_type1(img_features, train_newnet[list_relevant_indexes], train_newnet[list_indexes])
    else:
        new_weights = [1 for i in range(0,img_features.shape[0])]

    if num_rounds==1:
      old_weights = [1 for i in range(0,len(new_weights))]
      new_weights = [0.9*x[0] + 0.1*x[1] for x in zip(old_weights, new_weights)]
    else:
      file = open('saved_weights.txt','r')
      old_weights = file.read()
      old_weights = old_weights.split("\n")
      old_weights = [float(x) for x in range(0,len(old_weights))]
      file.close()

      new_weights = [0.9*x[0] + 0.1*x[1] for x in zip(old_weights, new_weights)]

    for i in range(0,similarity_df.shape[0]):
        similarity_df.loc[i,'Similarity'] = getMinkowskiSimilarity(img_features, train_newnet[i,:], new_weights)

    sorted_df = similarity_df.sort_values(by='Similarity', ascending=True)
    sorted_df['index_orig'] = sorted_df.index
    sorted_df = sorted_df.reset_index()

    img_paths = []
    img_paths_str = []
    img_dists = []
    img_indexes = []
    scores = []
    for i in range(0,6):
        tmp_path = Path("./static/img") / (sorted_df.loc[i,'filename'])
        tmp_dist = sorted_df.loc[i,'Similarity']
        tmp_index = sorted_df.loc[i,'index_orig']
        img_paths.append(tmp_path)
        img_paths_str.append(str(tmp_path))
        img_dists.append(tmp_dist)
        img_indexes.append(tmp_index)
        scores.append((sorted_df.loc[i,'Classes'], tmp_path))

    with open('paths_best.txt', 'w') as fp:
        for item in img_paths_str:
            fp.write("%s\n" % item)

    with open('indexes_best.txt', 'w') as fp:
        for item in img_indexes:
            fp.write("%s\n" % item)

    with open('saved_weights.txt', 'w') as fp:
        for item in new_weights:
            fp.write("%s\n" % item)

    with open('num_rounds.txt', 'w') as fp:
        fp.write("%s\n" % num_rounds)

    return render_template('index.html',
                            query_path=uploaded_img_path,
                            scores=scores)


if __name__=="__main__":
    app.run("0.0.0.0")