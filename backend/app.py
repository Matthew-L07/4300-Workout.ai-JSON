import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    exercises_df = pd.DataFrame(data['Exercises'])

app = Flask(__name__)
CORS(app)

def search_exercises(query):
    matches = exercises_df[exercises_df['Title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['Title', 'Desc', 'Rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

@app.route("/")
def home():
    return render_template('base.html', title="Fitness Search")

@app.route("/exercises")
def exercises_search():
    text = request.args.get("title")
    return search_exercises(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
