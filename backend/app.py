import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import yt_dlp


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

    for phrase, replacement in common_phrases.items():
        text = text.replace(phrase, replacement)

    text = re.sub(r'[^a-zA-Z0-9_\s]', '', text)

    return text


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def fix_typos(query_word, vocab, max_dist):
    fixed_typos = []
    for word in vocab:
        if abs(len(word) - len(query_word)) > max_dist:
            continue
        if edit_distance(query_word, word) <= max_dist:
            fixed_typos.append(word)
    return fixed_typos


os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    datalist = data["Exercises"]
    exercises_df = pd.DataFrame(datalist)

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

equipment_list = sorted(exercises_df['Equipment'].dropna().unique().tolist())
bodypart_list = sorted(exercises_df['BodyPart'].dropna().unique().tolist())
muscle_groups = [bp.lower() for bp in bodypart_list]

synonym_dict = {
    "arms": ["biceps", "triceps", "bis", "tris", "tri", "bi", "forearm"],
    "upper body": ["shoulders", "chest", "back", "lats", "laterals", "pecs", "abs", "abdomin", "core"],
    "core": ["abs", "abdominals", "midsection"],
    "legs": ["quads", "hamstrings", "glutes", "calves", "quadraceps"],
    "glutes": ["butt", "rear"],
    "cardio": ["endurance", "aerobic"],
}


descriptions = [preprocess_text((doc[0] + " ") * 10 + doc[1])
                for doc in documents]
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
bert_embeddings = bert_model.encode(descriptions, normalize_embeddings=True)
#vectorizer = TfidfVectorizer(stop_words='english')
#tfidf_matrix = vectorizer.fit_transform(descriptions)

vocab = set()
for desc in descriptions:
    for word in desc.split():
        vocab.add(word)

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template('base.html', title="Fitness Search",
                           equipment_list=equipment_list, bodypart_list=bodypart_list)


@app.route("/exercises")
def exercises_search():
    text = request.args.get("title", "")
    selected_equipment = request.args.get("equipment", "")
    selected_bodypart = request.args.get("bodypart", "").lower()

    query_tokens = preprocess_text(text).split()
    muscles_in_query = [t for t in query_tokens if t in muscle_groups]
    rel_query_tokens = [t for t in query_tokens if t not in muscles_in_query]

    filtered_indices = []
    for idx, doc in enumerate(documents):
        _, _, body_part, equip, _, _, _ = doc

        if selected_equipment and equip != selected_equipment:
            continue

        if selected_bodypart and body_part.lower() != selected_bodypart:
            continue

        if muscles_in_query and body_part.lower() not in muscles_in_query:
            continue

        filtered_indices.append(idx)

    if not filtered_indices:
        return jsonify([])

    query_words = []
    for token in rel_query_tokens:
        matched = fix_typos(token, vocab, max_dist=1)
        query_words.extend(matched if matched else [token])

    expanded_tokens = []
    for w in query_words:
        expanded_tokens.append(w)
        if w in synonym_dict:
            expanded_tokens.extend(synonym_dict[w])

    modified_query_text = ' '.join(expanded_tokens)

    # query_vector = vectorizer.transform([modified_query_text])
    # filtered_mat = tfidf_matrix[filtered_indices]
    # similarities = cosine_similarity(query_vector, filtered_mat).flatten()

    query_embedding = bert_model.encode(
        modified_query_text, normalize_embeddings=True)
    filtered_embeddings = np.array(
        [bert_embeddings[i] for i in filtered_indices])
    similarities = cosine_similarity(
        [query_embedding], filtered_embeddings).flatten()

    top_matches = similarities.argsort()[::-1][:10]
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
            exercise = {
                "Title": doc[0],
                "Desc": doc[1],
                "BodyPart": doc[2],
                "Equipment": doc[3],
                "Level": doc[4],
                "Rating": doc[5],
                "RatingDesc": doc[6],
                "TutorialURL": find_youtube_tutorial(doc[0]),
            }
            return render_template("exercise_detail.html", exercise=exercise)
    return "Exercise not found", 404


def find_youtube_tutorial(query):
    ydl_opts = {
        'quiet': True,
        'default_search': 'ytsearch1',
        'extract_flat': 'in_playlist',
        'force_generic_extractor': True,
        'match_filter': lambda info: (
            info.get('duration') <= 900 and
            "tutorial" in info.get('title', '').lower()
        )
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"{query}", download=False)
            if result.get('entries'):
                return result['entries'][0]['url']
    except Exception as e:
        print("Problem fetching video", e)

    return "https://www.youtube.com/results?search_query=" + '+'.join(query.split() + ["tutorial"])


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
