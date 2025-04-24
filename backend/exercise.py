import numpy as np
from flask import jsonify
from utils import preprocess_text

import numpy as np
from flask import jsonify
from utils import preprocess_text


def search_exercises(request, documents, query_embed, bert_embeddings, muscle_groups):
    text = request.args.get("title", "")
    selected_equipment = request.args.get("equipment", "")
    selected_bodypart = request.args.get("bodypart", "").lower()

    query_tokens = preprocess_text(text).split()
    filtered_indices = []
    rating_scores = []

    for idx, doc in enumerate(documents):
        _, _, body_part, equip, _, _, _, _ = doc
        if selected_equipment and equip != selected_equipment:
            continue
        if selected_bodypart and body_part.lower() != selected_bodypart:
            continue
        filtered_indices.append(idx)
        rating_scores.append(doc[5])

    if not filtered_indices:
        return jsonify([])

    query_embedding, _ = query_embed(text)
    bert_subset = [bert_embeddings[i] for i in filtered_indices]

    min_rating = min(rating_scores)
    max_rating = max(rating_scores)
    rating_range = max(max_rating - min_rating, 1e-6)
    rating_normalized = [
        (r - min_rating) / rating_range for r in rating_scores]
    scores = bert_subset @ query_embedding
    scores = 0.75 * scores + 0.25 * np.array(rating_normalized)
    top_matches = np.argsort(scores)[::-1][:10]

    query_word_embeds = {word: query_embed(word)[0] for word in query_tokens}

    results = []
    for idx in top_matches:
        doc = documents[filtered_indices[idx]]
        exercise_embed = bert_embeddings[filtered_indices[idx]]
        best_word = ""
        best_score = -1

        for word, emb in query_word_embeds.items():
            sim = np.dot(exercise_embed, emb)
            if sim > best_score:
                best_score = sim
                best_word = word

        explanation = f"This exercise matches well with the term '{best_word}' from your query."
        review = f"Reddit Based Review: {doc[6]}"

        exercise = {
            'Title': doc[0],
            'Desc': doc[1],
            'BodyPart': doc[2],
            'Equipment': doc[3],
            'Level': doc[4],
            'Rating': doc[5],
            'RatingDesc': doc[6],
            "FatigueLevel": doc[7]
        }

        results.append((exercise, review, explanation))

    return jsonify(results)