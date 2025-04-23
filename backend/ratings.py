import json
import math
import nltk
import numpy as np
from collections import defaultdict, Counter
from convokit import Corpus, download
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from app import preprocess_text, exercises_df

nltk.download("averaged_perceptron_tagger")
stop_words = set(stopwords.words("english"))

MAX_UTTERANCES = 10000
MAX_COMMENTS = 200
MIN_COMMENTS = 5
WEIGHT_CAP = 60
negation_window = 3
window_size = 3

positive_pivots = {"fun", "effective", "enjoy", "great", "strong", "amazing", "love", "rewarding", "motivating", "progress", "beneficial", "results"}
negative_pivots = {"pain", "injury", "hurt", "boring", "waste", "dangerous", "annoying", "hard", "difficult", "confusing", "fatigue", "bad", "inefficient"}
fatigue_pivots = {"exhausted", "drained", "tiring", "fatiguing", "killer", "brutal", "intense", "sweating", "hard", "challenging", "wiped", "taxing"}
negation_words = {"not", "no", "never", "none", "isnt", "wasnt", "arent", "dont", "didnt", "cant"}
excluded_words = {"pm", "etc", "id", "someone", "anyone", "add", "new", "see", "thanks", "foods", "recipes", "include", "offer", "managed", "rid", "search", "met", "app", "deadlifts", "elliptical"}
included_pivots = {"efficient", "rewarding", "motivating", "effective", "challenging", "results", "fun"}

subreddit = "Fitness"
corpus = Corpus(filename=download(f"subreddit-{subreddit}"))
sbert = SentenceTransformer("all-MiniLM-L6-v2")

ex_names = [preprocess_text(t).replace("_", " ").lower() for t in exercises_df["Title"]]
ex_embeddings = sbert.encode(ex_names, convert_to_tensor=True, batch_size=64)
ex_row_lookup = {preprocess_text(row["Title"]).replace("_", " ").lower(): row for _, row in exercises_df.iterrows()}

comment_texts, reddit_comments = [], []
for i, utt in enumerate(corpus.iter_utterances()):
    if i >= MAX_UTTERANCES:
        break
    text = utt.text
    if len(text) >= 20 and "http" not in text:
        reddit_comments.append(text)
        comment_texts.append(preprocess_text(text))

comment_embeddings = sbert.encode(comment_texts, convert_to_tensor=True, batch_size=64)
cos_sim = util.cos_sim(comment_embeddings, ex_embeddings)

matched_comments = defaultdict(list)
for i in range(len(comment_texts)):
    for j in range(len(ex_names)):
        if cos_sim[i][j] > 0.3:
            matched_ex = ex_names[j]
            if len(matched_comments[matched_ex]) < MAX_COMMENTS:
                matched_comments[matched_ex].append(reddit_comments[i])

co_occurrence = defaultdict(lambda: {"pos": 0, "neg": 0})
for text in reddit_comments:
    words = preprocess_text(text).split()
    for idx, word in enumerate(words):
        if word in stop_words or not word.isalpha():
            continue
        context = words[max(0, idx - window_size): idx] + words[idx + 1: idx + 1 + window_size]
        if set(context) & positive_pivots:
            co_occurrence[word]["pos"] += 1
        if set(context) & negative_pivots:
            co_occurrence[word]["neg"] += 1

word_scores = {w: c["pos"] / (c["pos"] + c["neg"]) for w, c in co_occurrence.items() if (c["pos"] + c["neg"]) > 1}

word_counts, raw_scores, raw_fatigue_scores = {}, {}, {}
for ex, comments in matched_comments.items():
    word_hits, score, fatigue_score = [], 0, 0
    for comment in comments:
        words = preprocess_text(comment).split()
        for i, word in enumerate(words):
            if word in fatigue_pivots:
                fatigue_score += 1
            if word in stop_words or not word.isalpha():
                continue
            is_negated = any(w in negation_words for w in words[max(0, i - negation_window):i])
            polarity = -1 if is_negated else 1
            if word in positive_pivots:
                score += 2.0 * polarity
            elif word in negative_pivots:
                score += -2.0 * polarity
            elif word_scores.get(word, 0.5) > 0.7:
                score += 1.0 * polarity
            elif word_scores.get(word, 0.5) < 0.3:
                score += -1.0 * polarity
            word_hits.append(word)
    raw_scores[ex] = score
    word_counts[ex] = word_hits
    raw_fatigue_scores[ex] = fatigue_score

scores = list(raw_scores.values())
mean = sum(scores) / len(scores) if scores else 0
std = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5 or 1e-10

def scaled_rating(score):
    z = (score - mean) / std
    sigmoid = 1 / (1 + math.exp(-2.5 * (z + 1.1)))
    return round(1.0 + sigmoid * 4.0, 2)

BODY_PART_FATIGUE = {
    "biceps": 1.0, "triceps": 1.0, "forearms": 1.0, "calves": 1.0,
    "shoulders": 2.5, "chest": 2.8, "abductors": 2.8, "adductors": 2.8, "traps": 2.8,
    "abdominals": 3.0, "middle back": 3.2, "lower back": 3.2, "lats": 3.2,
    "glutes": 4.0, "quadriceps": 4.0, "hamstrings": 4.5
}


fatigue_values = list(raw_fatigue_scores.values())
percentiles = np.percentile(fatigue_values, np.linspace(0, 100, 6)) if fatigue_values else [0, 1, 2, 3, 4, 5]

fatigue_values = list(raw_fatigue_scores.values())
if fatigue_values:
    percentiles = np.percentile(fatigue_values, [20, 40, 60, 80])
else:
    percentiles = [0, 1, 2, 3]

def scaled_fatigue_from_text(score):
    if score <= percentiles[0]:
        return 1.0
    elif score <= percentiles[1]:
        return 2.0
    elif score <= percentiles[2]:
        return 3.0
    elif score <= percentiles[3]:
        return 4.0
    else:
        return 5.0


def fatigue_from_bodypart(row, text_score=None):
    bp = str(row.get("BodyPart", "")).strip().lower()
    fatigue = BODY_PART_FATIGUE.get(bp)

    if fatigue is not None:
        return min(max(fatigue, 1.0), 5.0)

    if text_score is not None:
        return scaled_fatigue_from_text(text_score)

    return 3.0


word_freq = Counter()
for word_list in word_counts.values():
    word_freq.update(word_list)

def get_top_words(word_list, rating, count=5):
    local_freq = Counter(word_list)

    scored_pos, scored_neg = {}, {}
    for w, tf in local_freq.items():
        w = w.lower()
        if w in stop_words or w in excluded_words or not w.isalpha() or len(w) < 3:
            continue
        idf = math.log(1 + len(word_counts) / (1 + word_freq[w]))
        polarity_score = word_scores.get(w, 0.5)
        tfidf = tf * idf * (1.5 if w in included_pivots else 1.0)
        if polarity_score > 0.6:
            scored_pos[w] = tfidf
        elif polarity_score < 0.4:
            scored_neg[w] = tfidf

    top_pos = [w for w, _ in sorted(scored_pos.items(), key=lambda x: -x[1])]
    top_neg = [w for w, _ in sorted(scored_neg.items(), key=lambda x: -x[1])]
    if rating < 3.0:
        return top_neg[:count]
    elif rating < 3.75:
        half = count // 2
        return (top_pos[:count - half] + top_neg[:half])[:count]
    else:
        return top_pos[:count]

def basic_desc(rating):
    if rating >= 4.75:
        return "Users love this exercise!"
    elif rating >= 3.75:
        return "Most users like this exercise."
    elif rating >= 3.0:
        return "This exercise has mixed reactions."
    else:
        return "Most users do not prefer this exercise."

ratings_lookup = {}
for _, row in exercises_df.iterrows():
    title = preprocess_text(row["Title"]).replace("_", " ").lower()
    ex_key = title.replace(" ", "_").upper()
    matched_ex = title if title in matched_comments else None

    if not matched_ex or len(matched_comments[matched_ex]) < MIN_COMMENTS:
        rating, fatigue_level = 3.5, 3.0
        desc = "This is not a very popular exercise online."
    else:
        score = raw_scores[matched_ex]
        raw_rating = scaled_rating(score)
        popularity_score = min(len(matched_comments[matched_ex]), WEIGHT_CAP) / WEIGHT_CAP
        popularity_rating = 1.0 + 4.0 * popularity_score
        rating = round(0.75 * popularity_rating + 0.25 * raw_rating, 2)

        fatigue_raw = raw_fatigue_scores.get(matched_ex, 0)
        fatigue_level = fatigue_from_bodypart(ex_row_lookup[matched_ex], fatigue_raw)

        lead = basic_desc(rating)
        top_words = get_top_words(word_counts[matched_ex], rating)
        desc = f"{lead} People often mention: {', '.join(top_words)}."

    ratings_lookup[ex_key] = {
        "Rating": rating,
        "FatigueLevel": fatigue_level,
        "RatingDesc": desc
    }

with open("exercise_ratings.json", "w") as f:
    json.dump(ratings_lookup, f, indent=2)
    print("Finished uploading exercises!")