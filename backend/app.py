import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def preprocess_text(text):
    text = text.lower()
    
    phrase_replacements = {
        'push up': 'push_up',
        'pull up': 'pull_up',
        'skull crusher': 'skull_crusher',
        'sit up': 'sit_up',
        'leg raise': 'leg_raise'
    }
    
    for phrase, replacement in phrase_replacements.items():
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

def get_similar_words(query_word, vocab, max_dist):
    similar = []
    for word in vocab:
        if abs(len(word) - len(query_word)) > max_dist:
            continue
        if edit_distance(query_word, word) <= max_dist:
            similar.append(word)
    return similar

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


descriptions = [preprocess_text((doc[0] + " ") * 10 + doc[1]) for doc in documents]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(descriptions)

vocab = set()
for desc in descriptions:
    for word in desc.split():
        vocab.add(word)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html', title="Fitness Search",
                           equipment_list=equipment_list)

@app.route("/exercises")
def exercises_search():
    text = request.args.get("title", "")
    selected_equipment = request.args.get("equipment", "")

    query_tokens =  preprocess_text(text).split()

    query_muscle_groups = []
    rel_query_tokens = []

    for query_token in query_tokens:
        if query_token in muscle_groups:
            query_muscle_groups.append(query_token)
        else:
            rel_query_tokens.append(query_token)

    expanded_query_words = []
    for query_token in rel_query_tokens:
        matched = get_similar_words(query_token, vocab, max_dist=1)
        if matched:
            expanded_query_words.extend(matched)
        else:
            expanded_query_words.append(query_token) 

    modified_query_text = ' '.join(expanded_query_words)

    query_vector = vectorizer.transform([modified_query_text])

    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    sorted_indices = similarities.argsort()[::-1]

    top_matches = []
    for idx in sorted_indices:
        doc = documents[idx]
        title, desc, body_part, equip, lvl, rating, ratingdesc = doc

        if selected_equipment and equip != selected_equipment:
            continue

        if query_muscle_groups:
            if body_part.lower() not in query_muscle_groups:
                continue

        match = {
            'Title': title,
            'Desc': desc,
            'Rating': rating
        }
        top_matches.append(match)

        if len(top_matches) >= 10:
            break

    return jsonify(top_matches)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
