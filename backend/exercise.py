from rapidfuzz.distance import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_related_exercises(documents, bert_embeddings_np, main_exercises, count=3):
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

    main_indices = [i for i, doc in enumerate(
        documents) if doc[0] in main_titles]
    if not main_indices:
        return []

    mean_embed = np.mean(
        bert_embeddings_np[main_indices], axis=0, keepdims=True)
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