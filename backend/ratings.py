import re, json, math
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from convokit import Corpus, download
from app import preprocess_text, exercises_df

nltk.download("vader_lexicon")
nltk.download("stopwords")
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

MAX_COMMENTS = 50
MAX_UTTERANCES = 10000
subreddit = "weightroom" #Fitness for larger dataset

ex_names = {
    preprocess_text(e).replace("_", " ") 
    for e in exercises_df['Title']
    # if len(preprocess_text(e).split()) >= 2
}
ex_names = {
    e for e in ex_names 
    if not any(skip in e for skip in ["raise", "curl", "press", "row", "fly", "pull"])
}

name_to_ex = {e: re.compile(rf'\b{re.escape(e)}\b', re.IGNORECASE) for e in ex_names}
reddit_comments = {e: [] for e in ex_names}

keyword_to_ex = {}
for ex in ex_names:
    for word in ex.split():
        keyword_to_ex.setdefault(word, set()).add(ex)

all_ex = re.compile(r'\b(' + '|'.join(re.escape(e) for e in ex_names) + r')\b', re.IGNORECASE)

corpus = Corpus(filename=download(f"subreddit-{subreddit}"))
for i, utt in enumerate(corpus.iter_utterances()):
    if i >= MAX_UTTERANCES: break
    text = utt.text
    if len(text) < 20 or "http" in text or not all_ex.search(text):
        continue

    preprocess_words = set(preprocess_text(text).split())
    ex_matches = set()
    for word in preprocess_words:
        ex_matches.update(keyword_to_ex.get(word, set()))

    for ex in ex_matches:
        if len(reddit_comments[ex]) < MAX_COMMENTS and name_to_ex[ex].search(text):
            reddit_comments[ex].append(text)

    if i % 1000 == 0:
        print(f"Processed {i} utterances...")

def get_description(score):
    if score >= 0.5: return "People really enjoy this exercise and find it efficient!"
    elif score >= 0.2: return "This is an efficient exercise for working out."
    elif score >= 0.0: return "This is a pretty good exercise."
    elif score >= -0.2: return "This exercise might be more situational."
    return "This might not be worth your time or might risk injury."

raw_scores = {}
for _, row in exercises_df.iterrows():
    title = preprocess_text(row["Title"]).replace("_", " ").lower()
    fallback_text = f"{row['Title']} {row['Desc']}"
    comments = reddit_comments.get(title, [])
    
    if comments:
        score = round(sum(sia.polarity_scores(c)['compound'] for c in comments) / len(comments), 3)
    else:
        score = sia.polarity_scores(fallback_text)['compound']
    raw_scores[title] = score

scores = list(raw_scores.values())
mean = sum(scores) / len(scores)
std = (sum((x - mean)**2 for x in scores) / len(scores))**0.5 or 1e-10

def scaled_rating(score):
    z = (score - mean) / std
    sigmoid = 1 / (1 + math.exp(-2.5 * (z + 1.1)))
    return round(1.0 + sigmoid * 4.0, 1)  

ratings_lookup = {}
for _, row in exercises_df.iterrows():
    title = preprocess_text(row["Title"]).replace("_", " ").lower()
    ex_key = title.replace(" ", "_").upper()
    comments = reddit_comments.get(title, [])
    score = raw_scores[title]
    rating = scaled_rating(score)
    desc = get_description(score) 

    ratings_lookup[ex_key] = {"Rating": rating, "RatingDesc": desc}

with open("exercise_ratings.json", "w") as f:
    json.dump(ratings_lookup, f, indent=2)



