import re
from rapidfuzz.distance import Levenshtein
import yt_dlp

synonym_dict = {
    "arms": ["biceps", "triceps", "bis", "tris", "tri", "bi", "forearm"],
    "upper body": ["shoulders", "chest", "back", "lats", "laterals", "pecs", "abs", "abdomin", "core"],
    "core": ["abs", "abdominals"],
    "legs": ["quads", "hamstrings", "glutes", "calves", "quadraceps"],
    "glutes": ["butt", "rear"],
    "cardio": ["endurance", "aerobic"]
}

def preprocess_text(text):
    text = text.lower()
    common_phrases = {
        r'push[-\s]up(s?)': 'push_up',
        r'pu[-\s]up(s?)': 'push_up',
        r'skull[-\s]crusher(s?)': 'skull_crusher',
        r'sit[-\s]up(s?)': 'sit_up',
        r'leg[-\s]raise(s?)': 'leg_raise',
        r'dead[-\s]lift(s?)': 'deadlift',
        r'bench[-\s]press(es?)': 'bench_press'
    }
    for pattern, replacement in common_phrases.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r'[^a-zA-Z0-9_\s]', '', text)
    return text

def get_target_muscle_groups(query):
    tokens = preprocess_text(query).split()
    expanded = {word for token in tokens for word in synonym_dict.get(token, [token])}
    return list(expanded)

def fix_typos(query_word, vocab, max_dist):
    return [word for word in vocab if abs(len(word) - len(query_word)) <= max_dist and Levenshtein.distance(query_word, word) <= max_dist]

def correct_equipment(title, original, equipment_keywords):
    title_lower = title.lower()
    for keyword in equipment_keywords:
        if keyword in title_lower:
            return keyword.title()
    return original

def find_youtube_tutorial(query):
    ydl_opts = {
        'quiet': True,
        'default_search': 'ytsearch1',
        'extract_flat': 'in_playlist',
        'force_generic_extractor': True,
        'match_filter': lambda info: (
            info.get('duration') <= 900 and "tutorial" in info.get('title', '').lower()
        )
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"{query}", download=False)
            if result.get('entries'):
                return result['entries'][0]['url']
    except Exception as e:
        print("Problem fetching video", e)
    return "https://www.youtube.com/results?search_query=" + '+'.join(query.split() + ["tutorial"])

def create_query_embedder(vocab, model, synonym_dict):
    def query_embed(text):
        query_tokens = preprocess_text(text).split()
        corrected_tokens = []

        for token in query_tokens:
            matched = fix_typos(token, vocab, max_dist=1)
            corrected_tokens.extend(matched if matched else [token])

        expanded_tokens = corrected_tokens[:]
        for word in corrected_tokens:
            expanded_tokens.extend(synonym_dict.get(word, []))

        modified_query = ' '.join(expanded_tokens)
        query_embedding = model.encode(modified_query, normalize_embeddings=True)

        return query_embedding, expanded_tokens
    return query_embed

