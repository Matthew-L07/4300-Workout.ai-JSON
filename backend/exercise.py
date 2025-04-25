from rapidfuzz.distance import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_related_exercises(documents, bert_embeddings_np, main_exercises, count=3, bodypart_filter=None, used_titles=None):
    if not main_exercises:
        return []

    main_titles = {ex["Title"].upper() for ex in main_exercises}
    main_bodyparts = {ex["BodyPart"] for ex in main_exercises}
    title_lower = {title.lower() for title in main_titles}
    used_titles = set(title.upper() for title in used_titles or [])

    filtered_indices = [
        idx for idx, doc in enumerate(documents)
        if doc[0].upper() not in used_titles
        and (not bodypart_filter or doc[2].lower() == bodypart_filter.lower())
        and doc[2] in main_bodyparts
    ]

    if not filtered_indices:
        return []

    main_indices = [i for i, doc in enumerate(documents)
                    if doc[0].upper() in main_titles]
    if not main_indices:
        return []

    mean_embed = np.mean(
        bert_embeddings_np[main_indices], axis=0, keepdims=True)
    candidate_embeds = bert_embeddings_np[filtered_indices]
    if candidate_embeds.shape[0] == 0:
        return []

    sims = cosine_similarity(mean_embed, candidate_embeds).flatten()
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
