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
    filtered_words = [
        w for w in query_words if w not in generic_words] or query_words

    word_embeds = bert_model.encode(filtered_words, normalize_embeddings=True)
    sims = word_embeds @ exercise_embed
    return filtered_words[int(np.argmax(sims))]


def get_related_exercises(documents, bert_embeddings_np, query, count=3, bodypart_filter=None, used_titles=None, bert_model=None):
    used_titles = {title.upper() for title in (used_titles or [])}

    # Pre-filter documents based on used titles and bodypart
    filtered_docs = []
    filtered_embeds = []
    for idx, doc in enumerate(documents):
        if doc[0].upper() in used_titles:
            continue
        if bodypart_filter and doc[2].lower() != bodypart_filter.lower():
            continue
        filtered_docs.append(doc)
        filtered_embeds.append(bert_embeddings_np[idx])

    if not filtered_docs:
        return []

    # Process query once
    query_words = [w for w in preprocess_text(query).split(
    ) if w not in generic_words] or preprocess_text(query).split()
    query_embed = bert_model.encode(
        [" ".join(query_words)], normalize_embeddings=True)[0]

    # Calculate similarities in one operation
    sims = cosine_similarity([query_embed], filtered_embeds).flatten()

    # Get top results efficiently
    actual_count = min(count, len(sims))
    if actual_count == 0:
        return []

    top_indices = np.argsort(-sims)[:actual_count]

    # Build results without duplicate checks (already filtered)
    return [{
        'Title': filtered_docs[i][0],
        'Desc': filtered_docs[i][1],
        'BodyPart': filtered_docs[i][2],
        'Equipment': filtered_docs[i][3],
        'Level': filtered_docs[i][4],
        'Rating': filtered_docs[i][5],
        'RatingDesc': filtered_docs[i][6],
        'FatigueLevel': filtered_docs[i][7]
    } for i in top_indices]
