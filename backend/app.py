import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from utils import create_query_embedder, find_youtube_tutorial, preprocess_text, correct_equipment, get_target_muscle_groups, synonym_dict
from exercise import search_exercises
from routine import generate_workout_routine
from sentence_transformers import SentenceTransformer
from rapidfuzz.distance import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

current_directory = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_directory, 'init.json'), 'r') as f:
    data = json.load(f)["Exercises"]

with open("exercise_ratings.json", "r") as f:
    ratings_lookup = json.load(f)

exercises_df = pd.DataFrame(data)
exercises_df['Equipment'] = exercises_df['Equipment'].replace({'E-Z Curl Bar': 'Barbell', 'Medicine Ball': 'Other'})
equipment_list = sorted(exercises_df['Equipment'].dropna().unique())
equipment_keywords = [e.lower() for e in equipment_list if e not in {'Other', 'None'}]
bodypart_list = sorted(exercises_df['BodyPart'].dropna().unique())

documents = []
for ex in data:
    title_clean = preprocess_text(ex["Title"]).replace(" ", "_").upper()
    rating = ratings_lookup.get(title_clean, {}).get("Rating", ex["Rating"])
    rating_desc = ratings_lookup.get(title_clean, {}).get("RatingDesc", ex["RatingDesc"])
    fatigue_level = ratings_lookup.get(title_clean, {}).get("FatigueLevel", 3.0)
    documents.append((
        ex["Title"].upper(),
        ex["Desc"],
        ex["BodyPart"],
        correct_equipment(ex["Title"], ex["Equipment"], equipment_keywords),
        ex["Level"],
        rating,
        rating_desc,
        fatigue_level
    ))


descs = [preprocess_text(doc[1]) for doc in documents]
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
bert_embeddings = bert_model.encode(descs, normalize_embeddings=True)
bert_embeddings_np = np.array(bert_embeddings)
vocab = set(word for desc in descs for word in desc.split())

query_embed = create_query_embedder(vocab, bert_model, synonym_dict)

def get_related_exercises(main_exercises, count=3):
    if not main_exercises:
        return []

    main_titles = {ex["Title"].upper() for ex in main_exercises}
    main_bodyparts = {ex["BodyPart"] for ex in main_exercises}
    title_lower = {title.lower() for title in main_titles}

    filtered_indices = [
        idx for idx, doc in enumerate(documents)
        if doc[0] not in main_titles and
           doc[2] in main_bodyparts and
           doc[0].lower() not in title_lower and
           all(Levenshtein.distance(doc[0].lower(), t) >= 5 for t in title_lower)
    ]

    if not filtered_indices:
        return []

    main_indices = [i for i, doc in enumerate(documents) if doc[0] in main_titles]
    if not main_indices:
        return []

    mean_embed = np.mean(bert_embeddings_np[main_indices], axis=0, keepdims=True)
    candidate_embeds = bert_embeddings_np[filtered_indices]

    sims = cosine_similarity(mean_embed, candidate_embeds).flatten()
    top_indices = np.argpartition(-sims, count)[:count]

    results = []
    for i in top_indices:
        doc = documents[filtered_indices[i]]
        results.append({
            'Title': doc[0],
            'Desc': doc[1],
            'BodyPart': doc[2],
            'Equipment': doc[3],
            'Level': doc[4],
            'Rating': doc[5],
            'RatingDesc': doc[6],
            'FatigueLevel': doc[7]
        })

    return results

@app.route("/")
def home():
    return render_template("base.html", title="Fitness Search",
                           equipment_list=equipment_list,
                           bodypart_list=bodypart_list,
                           exercises=[])

@app.route("/exercises")
def exercises_search():
    return search_exercises(request, documents, query_embed, bert_embeddings, bodypart_list)

@app.route("/exercise/<title>")
def exercise_page(title):
    title = title.upper()
    for doc in documents:
        if doc[0] == title:
            return render_template("exercise_detail.html", exercise={
                "Title": doc[0],
                "Desc": doc[1],
                "BodyPart": doc[2],
                "Equipment": doc[3],
                "Level": doc[4],
                "Rating": doc[5],
                "RatingDesc": doc[6],
                "FatigueLevel": doc[7],
                "TutorialURL": find_youtube_tutorial(doc[0])
            })
    return "Exercise not found", 404

@app.route("/video/<title>")
def fetch_video(title):
    try:
        url = find_youtube_tutorial(title)
        return jsonify({"Video": url})
    except Exception as e:
        return jsonify({"Video": None, "Error": str(e)})

@app.route("/routine")
def workout_routine():
    query = request.args.get("title", "")
    selected_equipment = request.args.get("equipment", "")

    target_muscles_list = get_target_muscle_groups(query)
    target_muscles_str = " ".join(target_muscles_list) if target_muscles_list else query

    if not target_muscles_str:
        return jsonify({"main": [], "related": []})

    main_exercises = generate_workout_routine(target_muscles_str, selected_equipment, documents)
    related = get_related_exercises(main_exercises)

    main_wrapped = [(ex, f"This is a good {ex['BodyPart'].lower()} exercise.") for ex in main_exercises]
    related_wrapped = [(ex, f"This is a related {ex['BodyPart'].lower()} exercise.") for ex in related]

    return jsonify({
        "main": main_wrapped,
        "related": related_wrapped
    })

if __name__ == "__main__" and 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
