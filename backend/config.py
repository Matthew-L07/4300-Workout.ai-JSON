"""Configuration constants for the workout routine generator."""

# Similarity threshold for BERT model matching
SIMILARITY_THRESHOLD = 0.4

# Maximum number of exercises in a routine
MAX_EXERCISES = 5

# Target muscle group mappings for different workout types
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
