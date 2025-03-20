import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re
import math
import numpy as np


os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    datalist = data["Exercises"]
    exercises = pd.DataFrame(data['Exercises'])

    documents = [
        (
            pt["Title"].upper(),
            pt["Desc"],
            pt["BodyPart"],
            pt["Equipment"],
            pt["Level"],
            pt["Rating"],
            pt["RatingDesc"]
        )
        for pt in datalist
    ]

equipment_list = sorted(exercises['Equipment'].dropna().unique().tolist())
bodypart_list = sorted(exercises['BodyPart'].dropna().unique().tolist())
level_list = sorted(exercises['Level'].dropna().unique().tolist())

# vectorizer = TfidfVectorizer(stop_words='english')
# term_document_matrix = vectorizer.fit_transform(x[1] for x in documents)


# calculates df_count for each word in each description and maps unique words
word_to_index = {}
index_to_doc = {}
df = defaultdict(int)

processed_docs = []

iterations = 0
for i, d in enumerate(documents):
    desc = d[1].lower()
    words_in_desc = re.findall(r'\b[a-zA-Z0-9]+\b', desc)
    # unique_words = set(words_in_desc)

    for w in words_in_desc:
        if w not in word_to_index:
            word_to_index[w] = i
        df[w] += 1
    processed_docs.append(words_in_desc)
    index_to_doc[i] = d

total_docs = len(processed_docs)

idf_vector = {}
for word, count in df.items():
    idf_vector[word] = max(math.log(total_docs / count, 2), 0)


tf_idf_matrix = []

for words in processed_docs:
    tf = Counter(words)
    total_terms = len(words)
    tf_idf_vec = [0] * len(word_to_index)

    for word, freq in tf.items():
        if word in word_to_index:
            tf_idf_vec[word_to_index[word]] = (
                freq / total_terms) * idf_vector[word]

    tf_idf_matrix.append(tf_idf_vec)


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0

    return np.dot(vec1, vec2) / (np.linalg.norm(vec2) * np.linalg.norm(vec1))


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template('base.html', title="Fitness Search",
                           equipment_list=equipment_list,
                           bodypart_list=bodypart_list,
                           level_list=level_list)


@app.route("/exercises")
def exercises_search():
    text = request.args.get("title", "")
    equipment = request.args.get("equipment", "")
    bodypart = request.args.get("bodypart", "")
    level = request.args.get("level", "")

    text = text.strip().lower()
    query_words = re.findall(r'\b[\w-]+\b', text)
    query_tf = Counter(query_words)

    total_terms = len(query_words)
    query_vector = [0] * len(word_to_index)
    for word, freq in query_tf.items():
        if word in word_to_index:
            tf_weight = freq / total_terms
            query_vector[word_to_index[word]] = tf_weight * idf_vector[word]

    simularity = []
    for i, document_vector in enumerate(tf_idf_matrix):
        s = cosine_similarity(query_vector, document_vector)
        simularity.append((i, s))

    sim_sorted = sorted(simularity, key=lambda x: x[1], reverse=True)

    filtered_df = exercises
    top_matches = []

    sim_sorted = sorted(simularity, key=lambda x: x[1], reverse=True)

    top_matches = []

    for idx, _ in sim_sorted[:10]:
        doc = index_to_doc[idx]
        match = {
            'Title': doc[0],
            'Desc': doc[1],
            'Rating': doc[5]
        }
        top_matches.append(match)

    return top_matches

    # for idx, _ in sim_sorted[:10]:
    #     doc = index_to_doc[idx]
    #     title = doc[0]
    #     filtered_df = filtered_df[filtered_df['Title'].str.lower(
    #     ).str.contains(text.lower())]

    # matches_filtered = filtered_df[['Title', 'Desc', 'Rating']]
    # matches_filtered_json = matches_filtered.to_json(orient='records')
    # return matches_filtered_json

    # query = request.args.get("query", "").strip().lower()

    # if not query:
    #     return None

    # queryVec = vectorizer.transform([query])

    # score = cosine_similarity(queryVec, term_document_matrix).flatten()

    # topResults = score.argsort()[::-1][:10]

    # results = filtered_df.iloc[topResults][[
    #     'Title', 'Desc', 'BodyPart', 'Equipment', 'Level', 'Rating']].to_dict(orient='records')

    # return jsonify(results)

    if text:
        filtered_df = filtered_df[filtered_df['Title'].str.lower(
        ).str.contains(text.lower())]

    if equipment:
        filtered_df = filtered_df[filtered_df['Equipment'] == equipment]

    if bodypart:
        filtered_df = filtered_df[filtered_df['BodyPart'] == bodypart]

    if level:
        filtered_df = filtered_df[filtered_df['Level'] == level]

    matches_filtered = filtered_df[['Title', 'Desc', 'Rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
