import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    exercises_df = pd.DataFrame(data['Exercises'])

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
        for pt in exercises_df
    ]

    equipment_list = sorted(
        exercises_df['Equipment'].dropna().unique().tolist())
    bodypart_list = sorted(exercises_df['BodyPart'].dropna().unique().tolist())
    level_list = sorted(exercises_df['Level'].dropna().unique().tolist())

    vectorizer = TfidfVectorizer(stop_words='english')
    term_document_matrix = vectorizer.fit_transform(x[1] for x in documents)


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
    # text = request.args.get("title", "")
    # equipment = request.args.get("equipment", "")
    # bodypart = request.args.get("bodypart", "")
    # level = request.args.get("level", "")
    query = request.args.get("query", "").strip().lower()

    if not query:
        return None

    queryVec = vectorizer.transform([query])

    score = cosine_similarity(queryVec, term_document_matrix).flatten()

    topResults = score.argsort()[::-1][:10]

    filtered_df = exercises_df

    results = filtered_df.iloc[topResults][[
        'Title', 'Desc', 'BodyPart', 'Equipment', 'Level', 'Rating']].to_dict(orient='records')

    return jsonify(results)

    # if text:
    #     filtered_df = filtered_df[filtered_df['Title'].str.lower(
    #     ).str.contains(text.lower())]

    # if equipment:
    #     filtered_df = filtered_df[filtered_df['Equipment'] == equipment]

    # if bodypart:
    #     filtered_df = filtered_df[filtered_df['BodyPart'] == bodypart]

    # if level:
    #     filtered_df = filtered_df[filtered_df['Level'] == level]

    # matches_filtered = filtered_df[['Title', 'Desc', 'Rating']]
    # matches_filtered_json = matches_filtered.to_json(orient='records')
    # return matches_filtered_json


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
