"""
ocr_text_memory.py
==================
Accumulates and filters OCR text strings across frames for each entity.
Identifies the 'best' text over time by requiring strings to be seen multiple
times and choosing the longest stable string.
"""

import string
import logging
from typing import Dict, List, Set
from collections import defaultdict

log = logging.getLogger("ocr_text")

class OCRTextMemory:
    def __init__(self):
        # Maps entity_id -> dict mapping normalised_text -> count
        self._history: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Maps entity_id -> the original cased string for that normalised text
        self._original_case: Dict[str, Dict[str, str]] = defaultdict(dict)
        # Maps entity_id -> current best text
        self._best_text: Dict[str, str] = {}

    def _normalise(self, text: str) -> str:
        """Strip whitespace, punctuation, and convert to uppercase."""
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        clean = text.translate(translator).strip().upper()
        # Remove internal extra spaces
        return " ".join(clean.split())

    def _is_valid(self, text: str) -> bool:
        """Filter out obvious noise strings."""
        if len(text) < 3:
            return False
        # If it's purely numeric, ignore it (often just prices or capacities)
        if text.replace(" ", "").isdigit():
            return False
        return True

    def add_result(self, entity_id: str, texts: List[str]) -> bool:
        """
        Incorporate new text strings for an entity.
        Returns True if the 'best_text' changed as a result of this addition.
        """
        if not texts:
            return False

        changed = False
        for raw_text in texts:
            norm = self._normalise(raw_text)
            if self._is_valid(norm):
                self._history[entity_id][norm] += 1
                # Save the longest original casing seen for this normalisation
                if norm not in self._original_case[entity_id] or len(raw_text) > len(self._original_case[entity_id][norm]):
                    self._original_case[entity_id][norm] = raw_text

        return self._recalculate_best_text(entity_id)

    def _recalculate_best_text(self, entity_id: str) -> bool:
        """
        Determine the most reliable string seen so far.
        A string must be seen at least 2 times to be considered stable.
        Among stable strings, we generally prefer the longest one.
        """
        history = self._history.get(entity_id, {})
        
        # Filter to stable strings only (seen >= 2 times)
        stable_norms = [norm for norm, count in history.items() if count >= 2]
        
        if not stable_norms:
            return False
            
        # Sort by length (longest is usually the most complete read of a label)
        stable_norms.sort(key=len, reverse=True)
        
        best_norm = stable_norms[0]
        best_cased = self._original_case[entity_id][best_norm]
        
        current_best = self._best_text.get(entity_id, "")
        if current_best != best_cased:
            self._best_text[entity_id] = best_cased
            return True
            
        return False

    def get_best_text(self, entity_id: str) -> str:
        return self._best_text.get(entity_id, "")

    def get_all_texts(self, entity_id: str) -> List[str]:
        """Return the unique set of raw strings ever seen for this entity."""
        return list(self._original_case.get(entity_id, {}).values())

    def clear_entity(self, entity_id: str) -> None:
        """Free memory for an entity if it's completely forgotten (rare)."""
        self._history.pop(entity_id, None)
        self._original_case.pop(entity_id, None)
        self._best_text.pop(entity_id, None)
