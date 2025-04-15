import os
import re
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from rapidfuzz.distance import Levenshtein
import yt_dlp


app = Flask(__name__)
CORS(app)

current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)["Exercises"]
    exercises_df = pd.DataFrame(data)

updated_equipment = {'E-Z Curl Bar': 'Barbell', 'Medicine Ball': 'Other'}
exercises_df['Equipment'] = exercises_df['Equipment'].replace(
    updated_equipment)

equipment_list = sorted(exercises_df['Equipment'].dropna().unique())
equipment_keywords = [e.lower()
                      for e in equipment_list if e not in {'Other', 'None'}]
bodypart_list = sorted(exercises_df['BodyPart'].dropna().unique())
muscle_groups = [bp.lower() for bp in bodypart_list]


def preprocess_text(text):
    text = text.lower()
    common_phrases = {
        r'push[-\s]up(s?)': 'push_up',
        r'pu[-\s]up(s?)': 'push_up',
        r'skull[-\s]crusher(s?)': 'skull_crusher',
        r'sit[-\s]up(s?)': 'sit_up',
        r'leg[-\s]raise(s?)': 'leg_raise',
        r'dead[-\s]lift(s?)': 'deadlift',
        r'bench[-\s]press(es?)': 'bench_press'
    }
    for pattern, replacement in common_phrases.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r'[^a-zA-Z0-9_\s]', '', text)

    return text


def fix_typos(query_word, vocab, max_dist):
    fixed_typos = []
    for word in vocab:
        if abs(len(word) - len(query_word)) > max_dist:
            continue
        if Levenshtein.distance(query_word, word) <= max_dist:
            fixed_typos.append(word)

    return fixed_typos


def correct_equipment(title, original):
    title_lower = title.lower()
    for keyword in equipment_keywords:
        if keyword in title_lower:
            return keyword.title()
    return original


synonym_dict = {
    "arms": ["biceps", "triceps", "bis", "tris", "tri", "bi", "forearm"],
    "upper body": ["shoulders", "chest", "back", "lats", "laterals", "pecs", "abs", "abdomin", "core"],
    "core": ["abs", "abdominals"],
    "legs": ["quads", "hamstrings", "glutes", "calves", "quadraceps"],
    "glutes": ["butt", "rear"],
    "cardio": ["endurance", "aerobic"]
}


with open("exercise_ratings.json", "r") as f:
    ratings_lookup = json.load(f)

documents = [
    (
        pt["Title"].upper(),
        pt["Desc"],
        pt["BodyPart"],
        correct_equipment(pt["Title"], pt["Equipment"]),
        pt["Level"],
        ratings_lookup.get(preprocess_text(pt["Title"]).replace(
            " ", "_").upper(), {}).get("Rating", pt["Rating"]),
        ratings_lookup.get(preprocess_text(pt["Title"]).replace(
            " ", "_").upper(), {}).get("RatingDesc", pt["RatingDesc"])
    )
    for pt in data
]

descs = [preprocess_text(doc[1]) for doc in documents]
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
bert_embeddings = bert_model.encode(descs, normalize_embeddings=True)

vocab = set(word for desc in descs for word in desc.split())


@app.route("/")
def home():
    return render_template("base.html", title="Fitness Search",
                           equipment_list=equipment_list,
                           bodypart_list=bodypart_list)


@app.route("/exercises")
def exercises_search():
    text = request.args.get("title", "")
    selected_equipment = request.args.get("equipment", "")
    selected_bodypart = request.args.get("bodypart", "").lower()

    query_tokens = preprocess_text(text).split()
    muscle_terms = [t for t in query_tokens if t in muscle_groups]
    nonmuscle_terms = [t for t in query_tokens if t not in muscle_terms]

    filtered_indices = []
    rating_scores = []
    for idx, doc in enumerate(documents):
        _, _, body_part, equip, _, _, _ = doc
        if selected_equipment and equip != selected_equipment:
            continue
        if selected_bodypart and body_part.lower() != selected_bodypart:
            continue
        if muscle_terms and body_part.lower() not in muscle_terms:
            continue

        filtered_indices.append(idx)

        _, _, _, _, _, rating, _ = documents[idx]
        rating_scores.append(rating)

    if not filtered_indices:
        return jsonify([])

    corrected_tokens = []
    for token in nonmuscle_terms:
        matched = fix_typos(token, vocab, max_dist=1)
        corrected_tokens.extend(matched if matched else [token])

    expanded_tokens = corrected_tokens[:]
    for word in corrected_tokens:
        expanded_tokens.extend(synonym_dict.get(word, []))

    modified_query = ' '.join(expanded_tokens)

    query_embeddings = bert_model.encode(
        modified_query, normalize_embeddings=True)

    scores = bert_embeddings[filtered_indices] @ query_embeddings

    min_rating = min(rating_scores)
    max_rating = max(rating_scores)
    rating_range = max(max_rating - min_rating, 1e-6)

    rating_normalized = [
        (r - min_rating) / rating_range for r in rating_scores]
    scores = 0.75 * scores + 0.25 * np.array(rating_normalized)
    top_matches = np.argsort(scores)[::-1][:10]

    results = []

    for idx in top_matches:
        original_idx = filtered_indices[idx]
        title, desc, _, _, _, rating, _ = documents[original_idx]
        results.append({
            'Title': title,
            'Desc': desc,
            'Rating': rating
        })

    return jsonify(results)


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
                "TutorialURL": find_youtube_tutorial(doc[0])
            })
    return "Exercise not found", 404


def find_youtube_tutorial(query):

    ydl_opts = {
        'quiet': True,
        'default_search': 'ytsearch1',
        'extract_flat': 'in_playlist',
        'force_generic_extractor': True,
        'match_filter': lambda info: (
            info.get('duration') <= 900 and "tutorial" in info.get(
                'title', '').lower()
        )
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"{query}", download=False)
            if result.get('entries'):
                url = result['entries'][0]['url']
                return url
    except Exception as e:
        print("Problem fetching video", e)

    default_url = "https://www.youtube.com/results?search_query=" + \
        '+'.join(query.split() + ["tutorial"])
    return default_url


if __name__ == "__main__" and 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
