import ast
import re
from typing import List


POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "love",
    "like",
    "happy",
    "positive",
}

NEGATIVE_WORDS = {
    "bad",
    "worse",
    "worst",
    "poor",
    "hate",
    "sad",
    "negative",
}


def parse_comments(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item is not None]
    return []


def sentiment_score(text) -> float:
    if not text:
        return 0.0
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_]+", str(text).lower())
    if not tokens:
        return 0.0
    positive = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    return float(positive - negative) / float(len(tokens))
