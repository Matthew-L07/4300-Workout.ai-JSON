"""Workout routine generation module using BERT embeddings for exercise matching."""

import logging
from collections import Counter
from typing import List, Dict, Set, Optional, Tuple, Any
import random
import numpy as np
from app import preprocess_text
from config import TARGET_MAPPINGS, SIMILARITY_THRESHOLD, MAX_EXERCISES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for BERT embeddings of keywords
_keyword_embeddings_cache: Dict[str, np.ndarray] = {}


def get_best_mapping_keyword(query: str, bert_model: Any) -> str:
    """
    Find the best matching keyword from TARGET_MAPPINGS using BERT embeddings.

    Args:
        query: The search query string
        bert_model: The BERT model for generating embeddings

    Returns:
        The best matching keyword from TARGET_MAPPINGS
    """
    if not query or not isinstance(query, str):
        logger.warning("Invalid query input in get_best_mapping_keyword")
        return "full body"  # Default fallback

    query_clean = preprocess_text(query)
    query_embed = bert_model.encode(
        [query_clean], normalize_embeddings=True)[0]

    # Use cached embeddings if available
    keywords = list(TARGET_MAPPINGS.keys())
    if tuple(keywords) not in _keyword_embeddings_cache:
        _keyword_embeddings_cache[tuple(keywords)] = bert_model.encode(
            keywords, normalize_embeddings=True
        )

    keyword_embeds = _keyword_embeddings_cache[tuple(keywords)]
    sims = keyword_embeds @ query_embed
    best_idx = int(np.argmax(sims))
    return keywords[best_idx]


def get_targets(query: str, bert_model: Optional[Any] = None,
                explicit_bodypart: Optional[str] = None) -> List[str]:
    """
    Get target muscle groups based on the query and optional body part filter.

    Args:
        query: The search query string
        bert_model: Optional BERT model for semantic matching
        explicit_bodypart: Optional specific body part to include

    Returns:
        List of target muscle groups
    """
    if not query or not isinstance(query, str):
        logger.warning("Invalid query input in get_targets")
        return ["Quadriceps", "Chest", "Back", "Shoulders"]  # Default fallback

    query = str(query).lower()
    result = []

    # First try exact keyword matching
    for keyword, targets in TARGET_MAPPINGS.items():
        if keyword in query:
            result = targets
            break

    # If no exact match and BERT model available, try semantic matching
    if not result and bert_model:
        try:
            query_clean = preprocess_text(query)
            query_embed = bert_model.encode(
                [query_clean], normalize_embeddings=True)[0]

            keywords = list(TARGET_MAPPINGS.keys())
            if tuple(keywords) not in _keyword_embeddings_cache:
                _keyword_embeddings_cache[tuple(keywords)] = bert_model.encode(
                    keywords, normalize_embeddings=True
                )

            keyword_embeds = _keyword_embeddings_cache[tuple(keywords)]
            sims = keyword_embeds @ query_embed
            best_idx = int(np.argmax(sims))
            best_score = sims[best_idx]

            if best_score >= SIMILARITY_THRESHOLD:
                result = TARGET_MAPPINGS[keywords[best_idx]]
                logger.info(
                    f"Found semantic match with score {best_score:.2f}")
        except Exception as e:
            logger.error(f"Error in BERT matching: {str(e)}")

    # Fallback to basic body part matching
    if not result:
        if "upper" in query:
            result = ["Biceps", "Triceps", "Shoulders", "Chest"]
        elif "lower" in query:
            result = ["Quadriceps", "Hamstrings", "Glutes", "Calves"]
        else:
            result = ["Quadriceps", "Chest", "Back", "Shoulders"]

    # Apply explicit body part filter if provided
    if explicit_bodypart:
        result = [explicit_bodypart] + \
            [r for r in result if r != explicit_bodypart]

    # Ensure unique and correct number of targets
    result = list(dict.fromkeys(result))
    while len(result) < 4:
        result.append(random.choice(result))
    if len(result) > 4:
        result = result[:4]

    return result


def generate_workout_routine(
    query: str,
    selected_equipment: Optional[str] = None,
    documents: Optional[List[Tuple]] = None,
    bodypart_filter: Optional[str] = None,
    used_exercises: Set[str] = set(),
    bert_model: Optional[Any] = None,
    bert_embeddings: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Generate a workout routine based on the query and available exercises.

    Args:
        query: The search query string
        selected_equipment: Optional equipment filter
        documents: List of exercise documents
        bodypart_filter: Optional specific body part to target
        used_exercises: Set of already used exercise names
        bert_model: BERT model for semantic matching
        bert_embeddings: Pre-computed BERT embeddings for exercises

    Returns:
        List of exercise dictionaries forming the routine
    """
    # Input validation
    if not query or not isinstance(query, str):
        logger.warning("Invalid query input in generate_workout_routine")
        return []

    if not documents or not isinstance(documents, list):
        logger.warning("Invalid documents input in generate_workout_routine")
        return []

    if bert_model is None or bert_embeddings is None:
        logger.warning(
            "Missing BERT model or embeddings in generate_workout_routine")
        return []

    try:
        targets = get_targets(query, bert_model=bert_model,
                              explicit_bodypart=bodypart_filter)
        used_set = set(used_exercises)
        target_counts = Counter(targets)

        # Generate query embedding
        query_embed = bert_model.encode(
            [preprocess_text(query)], normalize_embeddings=True)[0]

        # Filter and score candidates in one pass using numpy operations
        candidate_indices = []
        candidate_docs = []

        for idx, doc in enumerate(documents):
            if (doc[0] not in used_set and
                (not selected_equipment or doc[3] == selected_equipment) and
                    doc[2] in target_counts):
                candidate_indices.append(idx)
                candidate_docs.append(doc)

        if len(candidate_indices) < MAX_EXERCISES:
            # Relax the body part filter if we don't have enough candidates
            for idx, doc in enumerate(documents):
                if (doc[0] not in used_set and
                        (not selected_equipment or doc[3] == selected_equipment)):
                    if idx not in candidate_indices:
                        candidate_indices.append(idx)
                        candidate_docs.append(doc)

        if not candidate_indices:
            logger.warning("No valid candidates found for routine generation")
            return []

        # Compute similarities using numpy vectorized operations
        candidate_embeddings = bert_embeddings[candidate_indices]
        sims = np.dot(candidate_embeddings, query_embed)

        # Combine similarities with ratings for sorting
        sims_with_ratings = np.column_stack((
            sims,
            [doc[5] for doc in candidate_docs]  # Ratings
        ))

        # Sort by similarity and rating
        sorted_indices = np.lexsort(
            (sims_with_ratings[:, 1], sims_with_ratings[:, 0]))[::-1]

        # Build routine respecting target counts
        used_per_bodypart = Counter()
        routine = []

        for idx in sorted_indices:
            doc = candidate_docs[idx]
            part = doc[2]
            if used_per_bodypart[part] < target_counts[part]:
                routine.append(doc)
                used_per_bodypart[part] += 1
            if len(routine) == MAX_EXERCISES:
                break

        # Fill remaining slots if needed
        if len(routine) < MAX_EXERCISES:
            for idx in sorted_indices:
                doc = candidate_docs[idx]
                if doc not in routine:
                    routine.append(doc)
                if len(routine) == MAX_EXERCISES:
                    break

        # Sort by fatigue level
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

    except Exception as e:
        logger.error(f"Error generating workout routine: {str(e)}")
        return []
