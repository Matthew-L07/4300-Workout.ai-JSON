from utils import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

generic_words = stop_words | {
    "body", "physique", "exercise", "routine", "program", "workout",
    "want", "need", "feel", "goal", "make", "get", "do", "done"
}

def get_best_matching_word(query, exercise_embed, bert_model):
    query_words = preprocess_text(query).split()
    filtered_words = [w for w in query_words if w not in generic_words]
    
    if not filtered_words:
        filtered_words = query_words

    word_embeds = bert_model.encode(filtered_words, normalize_embeddings=True)
    sims = word_embeds @ exercise_embed
    best_idx = int(np.argmax(sims))
    return filtered_words[best_idx]


def get_related_exercises(documents, bert_embeddings_np, query, count=3, bodypart_filter=None, used_titles=None, bert_model=None):
    used_titles = set(title.upper() for title in used_titles or [])

    filtered_indices = [
        idx for idx, doc in enumerate(documents)
        if doc[0].upper() not in used_titles
        and (not bodypart_filter or doc[2].lower() == bodypart_filter.lower())
    ]

    if not filtered_indices:
        return []

    query_words = [w for w in preprocess_text(query).split() if w not in generic_words]
    if not query_words:
        query_words = preprocess_text(query).split()

    query_embed = bert_model.encode([" ".join(query_words)], normalize_embeddings=True)[0]

    candidate_embeds = bert_embeddings_np[filtered_indices]
    sims = cosine_similarity([query_embed], candidate_embeds).flatten()

    actual_count = min(count, len(sims))
    if actual_count == 0:
        return []

    top_indices = np.argsort(-sims)[:actual_count]

    seen_titles = set(used_titles)
    results = []
    for i in top_indices:
        doc = documents[filtered_indices[i]]
        title_upper = doc[0].upper()
        if title_upper in seen_titles:
            continue
        seen_titles.add(title_upper)
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
        if len(results) == count:
            break

    return results

