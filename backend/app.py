import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from utils import create_query_embedder, find_youtube_tutorial, preprocess_text, correct_equipment, synonym_dict
from exercise import get_related_exercises
from routine import generate_workout_routine, get_targets
from sentence_transformers import SentenceTransformer
import numpy as np
from rapidfuzz.distance import Levenshtein


app = Flask(__name__)
CORS(app)

current_directory = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_directory, 'init.json'), 'r') as f:
    data = json.load(f)["Exercises"]

with open("exercise_ratings.json", "r") as f:
    ratings_lookup = json.load(f)

exercises_df = pd.DataFrame(data)
exercises_df['Equipment'] = exercises_df['Equipment'].replace(
    {'E-Z Curl Bar': 'Barbell', 'Medicine Ball': 'Other'})
equipment_list = sorted(exercises_df['Equipment'].dropna().unique())
equipment_keywords = [e.lower()
                      for e in equipment_list if e not in {'Other', 'None'}]
bodypart_list = sorted(exercises_df['BodyPart'].dropna().unique())

documents = []
for ex in data:
    title_clean = preprocess_text(ex["Title"]).replace(" ", "_").upper()
    rating = ratings_lookup.get(title_clean, {}).get("Rating", ex["Rating"])
    rating_desc = ratings_lookup.get(title_clean, {}).get(
        "RatingDesc", ex["RatingDesc"])
    fatigue_level = ratings_lookup.get(
        title_clean, {}).get("FatigueLevel", 3.0)
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

@app.route("/")
def home():
    return render_template("base.html", title="Fitness Search",
                           equipment_list=equipment_list,
                           bodypart_list=bodypart_list,
                           exercises=[])


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
    selected_bodypart = request.args.get("bodypart", "")

    target_muscles_list = get_targets(query, explicit_bodypart=selected_bodypart)
    target_muscles_str = " ".join(target_muscles_list)

    if not target_muscles_str:
        return jsonify({"main": [], "related": []})

    main_exercises = generate_workout_routine(
        target_muscles_str,
        selected_equipment,
        documents,
        bodypart_filter=selected_bodypart        
    )

    related_exercises = get_related_exercises(documents, bert_embeddings_np, main_exercises, count=8)

    combined = []
    seen_titles = set()

    for ex in main_exercises + related_exercises:
        title = ex["Title"]
        if title not in seen_titles:
            combined.append(ex)
            seen_titles.add(title)
        if len(combined) >= 8:
            break

    full_routine = combined

    if hasattr(search_result, "json"):
        search_result = search_result.json

    def is_similar(title1, title2, max_dist=5):
        return Levenshtein.distance(title1.lower(), title2.lower()) < max_dist

    additional = []
    for r in search_result:
        new_title = r[0]["Title"]
        if not any(is_similar(new_title, existing["Title"]) for existing in full_routine):
            additional.append(r)
        if len(additional) >= 5:
            break    
    main_wrapped = [
        (ex, f"Reddit Review: {ex['RatingDesc']}", f"This is a good {ex['BodyPart'].lower()} exercise.")
        for ex in full_routine
    ]

    related_wrapped = [
        (ex[0], ex[1], ex[2]) for ex in additional
    ] if additional else []

    return jsonify({
        "main": main_wrapped,
        "related": related_wrapped
    })


if __name__ == "__main__" and 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)