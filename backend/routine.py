import random
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess_text, synonym_dict

def get_target_muscle_groups(query):
    tokens = preprocess_text(query).split()
    expanded = {word for token in tokens for word in synonym_dict.get(token, [token])}
    return list(expanded)

def generate_workout_routine(query, selected_equipment, documents, query_embed, bert_embeddings):
    query_embedding, expanded_tokens = query_embed(query)

    scores = []
    for i, doc in enumerate(documents):
        if selected_equipment and doc[3] != selected_equipment:
            continue

        bodypart = doc[2].lower()
        if any(token in bodypart for token in expanded_tokens):
            sim = cosine_similarity([query_embedding], [bert_embeddings[i]])[0][0]
            scores.append((sim, i))

    if not scores:
        return []

    scores.sort(reverse=True)
    top_indices = [i for _, i in scores[:4]]

    routine = []
    for i in top_indices:
        doc = documents[i]
        routine.append({
            "Title": doc[0],
            "Desc": doc[1],
            "BodyPart": doc[2],
            "Equipment": doc[3],
            "Level": doc[4],
            "Rating": doc[5],
            "RatingDesc": doc[6]
        })

    return routine

