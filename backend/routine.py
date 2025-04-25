from collections import Counter
import random
import numpy as np
from app import preprocess_text

TARGET_MAPPINGS = {
    "vertical": ["Quadriceps", "Calves", "Glutes", "Hamstrings"], 
    "jump": ["Quadriceps", "Glutes", "Hamstrings", "Calves"],
    "running": ["Hamstrings", "Calves", "Glutes", "Quadriceps"], 
    "sprint": ["Hamstrings", "Calves", "Glutes", "Abdominals"],
    "speed": ["Hamstrings", "Calves", "Glutes", "Quadriceps"],
    "agility": ["Calves", "Quadriceps", "Adductors", "Abductors"],

    "beach body": ["Abdominals", "Chest", "Shoulders", "Quadriceps"],
    "lean": ["Abdominals", "Calves", "Shoulders", "Quadriceps"],
    "cut": ["Abdominals", "Abdominals", "Shoulders", "Calves"],
    "aesthetic": ["Chest", "Shoulders", "Abdominals", "Biceps"],

    "bulk": ["Chest", "Back", "Quadriceps", "Shoulders"],
    "mass": ["Chest", "Back", "Hamstrings", "Glutes"],
    "gain": ["Quadriceps", "Chest", "Back", "Shoulders"],

    "fat loss": ["Abdominals", "Quadriceps", "Shoulders", "Calves"],
    "hiit": ["Quadriceps", "Chest", "Back", "Abdominals", "Calves", "Shoulders"],
    "conditioning": ["Abdominals", "Calves", "Hamstrings", "Shoulders"],
    "cardio": ["Quadriceps", "Calves", "Hamstrings", "Shoulders", "Abdominals"],

    "push day": ["Chest", "Triceps", "Shoulders", "Chest"],
    "pull day": ["Lats", "Biceps", "Middle Back", "Lower Back"],
    "full body": ["Quadriceps", "Hamstrings", "Chest", "Lats", "Shoulders", "Glutes", "Abdominals", "Calves"],
    "strength": ["Quadriceps", "Hamstrings", "Lats", "Chest", "Shoulders", "Glutes"],
    "crossfit": ["Quadriceps", "Hamstrings", "Chest", "Lats", "Shoulders", "Abdominals", "Glutes", "Calves"],

    "arms": ["Biceps", "Triceps", "Forearms", "Biceps"],
    "biceps": ["Biceps", "Forearms", "Shoulders", "Lats"],
    "triceps": ["Triceps", "Chest", "Shoulders", "Forearms"],
    "shoulders": ["Shoulders", "Traps", "Chest", "Triceps"],
    "chest": ["Chest", "Shoulders", "Triceps", "Biceps"],
    "back": ["Lats", "Traps", "Middle Back", "Lower Back", "Biceps"],
    "abs": ["Abdominals", "Abdominals", "Lower Back"],
    "core": ["Abdominals", "Lower Back", "Abdominals", "Glutes"],
    "legs": ["Quadriceps", "Hamstrings", "Calves", "Glutes", "Adductors", "Abductors"],
    "glutes": ["Glutes", "Hamstrings", "Quadriceps", "Calves"],
    "thighs": ["Quadriceps", "Hamstrings", "Adductors", "Glutes"],
    "calves": ["Calves", "Hamstrings", "Glutes", "Quadriceps"],
    "butt": ["Glutes", "Hamstrings", "Quadriceps", "Abductors"]
}



def get_targets(query, bert_model=None, explicit_bodypart=None):
    query = str(query).lower()
    result = []

    for keyword, targets in TARGET_MAPPINGS.items():
        if keyword in query:
            result = targets
            break

    if not result and bert_model:
        query_clean = preprocess_text(query)
        query_embed = bert_model.encode([query_clean], normalize_embeddings=True)[0]

        keywords = list(TARGET_MAPPINGS.keys())
        keyword_embeds = bert_model.encode(keywords, normalize_embeddings=True)
        sims = keyword_embeds @ query_embed
        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]

        if best_score >= 0.4:
            result = TARGET_MAPPINGS[keywords[best_idx]]

    if not result:
        if "upper" in query:
            result = ["Biceps", "Triceps", "Shoulders", "Chest"]
        elif "lower" in query:
            result = ["Quadriceps", "Hamstrings", "Glutes", "Calves"]
        else:
            result = ["Quadriceps", "Chest", "Back", "Shoulders"]

    if explicit_bodypart:
        result = [explicit_bodypart] + [r for r in result if r != explicit_bodypart]

    result = list(dict.fromkeys(result))  

    while len(result) < 4:
        result.append(random.choice(result))
    if len(result) > 4:
        result = result[:4]

    return result


def generate_workout_routine(query, selected_equipment=None, documents=None,
                              bodypart_filter=None, used_exercises=set(),
                              bert_model=None, bert_embeddings=None):
    targets = get_targets(query, bert_model=bert_model, explicit_bodypart=bodypart_filter)
    used_set = set(used_exercises)
    target_counts = Counter(targets)

    if not documents or bert_model is None or bert_embeddings is None:
        return []

    query_embed = bert_model.encode([preprocess_text(query)], normalize_embeddings=True)[0]

    candidates = [
        (idx, doc) for idx, doc in enumerate(documents)
        if doc[0] not in used_set and
           (not selected_equipment or doc[3] == selected_equipment) and
           doc[2] in target_counts
    ]

    if len(candidates) < 5:
        candidates = [
            (idx, doc) for idx, doc in enumerate(documents)
            if doc[0] not in used_set and
               (not selected_equipment or doc[3] == selected_equipment)
        ]

    if not candidates:
        return []

    sims = []
    for idx, doc in candidates:
        sim = np.dot(query_embed, bert_embeddings[idx])
        sims.append((sim, doc))

    sims_sorted = sorted(sims, key=lambda x: (x[0], x[1][5]), reverse=True)

    used_per_bodypart = Counter()
    routine = []

    for _, doc in sims_sorted:
        part = doc[2]
        if used_per_bodypart[part] < target_counts[part]:
            routine.append(doc)
            used_per_bodypart[part] += 1
        if len(routine) == 5:
            break

    if len(routine) < 5:
        for _, doc in sims_sorted:
            if doc not in routine:
                routine.append(doc)
            if len(routine) == 5:
                break

    routine_sorted = sorted(routine, key=lambda doc: doc[7], reverse=True)

    return [
        {
            "Title": doc[0],
            "Desc": doc[1],
            "BodyPart": doc[2],
            "Equipment": doc[3],
            "Level": doc[4],
            "Rating": doc[5],
            "RatingDesc": doc[6],
            "FatigueLevel": doc[7]
        }
        for doc in routine_sorted
    ]
