import random

TARGET_MAPPINGS = {
    "butt": ["Glutes"],
    "glutes": ["Glutes"],
    "legs": ["Quadriceps", "Hamstrings", "Calves", "Adductors", "Abductors"],
    "thighs": ["Quadriceps", "Hamstrings"],
    "calves": ["Calves"],
    
    "arms": ["Biceps", "Triceps", "Forearms"],
    "biceps": ["Biceps"],
    "triceps": ["Triceps"],
    "shoulders": ["Shoulders"],
    "chest": ["Chest"],
    "back": ["Lats", "Traps", "Middle Back", "Lower Back"],
    
    "abs": ["Abdominals"],
    "core": ["Abdominals", "Lower Back"],
    
    "crossfit": ["Quadriceps", "Chest", "Back", "Shoulders"],
    "hiit": ["Quadriceps", "Chest", "Back", "Shoulders"],
    "cardio": ["Quadriceps", "Chest", "Back", "Shoulders"],
    "strength": ["Quadriceps", "Chest", "Back", "Shoulders"],
    "full body": ["Quadriceps", "Chest", "Back", "Shoulders"]
}

def get_targets(query):
    """Map user query to target muscle groups using exact labels from data"""
    query = str(query).lower()
    
    for keyword, targets in TARGET_MAPPINGS.items():
        if keyword in query:
            return targets
            
    if "upper" in query:
        return ["Biceps", "Triceps", "Shoulders", "Chest"]
    if "lower" in query:
        return ["Quadriceps", "Hamstrings", "Glutes", "Calves"]
    
    return ["Quadriceps", "Chest", "Back", "Shoulders"]

def generate_workout_routine(query, selected_equipment=None, documents=None, used_exercises=set()):
    targets = set(get_targets(query))
    used_set = set(used_exercises)

    if not documents:
        return []

    ex_in_muscle_group = [
        doc for doc in documents
        if doc[0] not in used_set and
           (not selected_equipment or doc[3] == selected_equipment) and
           doc[2] in targets
    ]

    if len(ex_in_muscle_group) < 4:
        ex_in_muscle_group = [
            doc for doc in documents
            if doc[0] not in used_set and
               (not selected_equipment or doc[3] == selected_equipment)
        ]

    if not ex_in_muscle_group:
        return []

    random.shuffle(ex_in_muscle_group)
    selected = ex_in_muscle_group[:4]

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
        for doc in selected
    ]
