"""
query_interpreter.py
====================
Translates natural language queries into structured dictionary filters for the SearchEngine.
Uses rule-based NLP (regex + keywords) without needing an LLM.
"""

import re
import logging
from product_knowledge import PRODUCT_KB

log = logging.getLogger("query_interpreter")

# Extract unique valid brands and product types from knowledge base
VALID_BRANDS = {v["brand"].lower(): v["brand"] for v in PRODUCT_KB.values()}
VALID_PRODUCT_TYPES = {v["product_type"].lower(): v["product_type"] for v in PRODUCT_KB.values()}

# Common YOLO Categories
VALID_CATEGORIES = {
    "person", "car", "bottle", "cup", "laptop", "book", "chair", "tv", "cell phone", "keyboard", "mouse"
}

# Events
EVENT_KEYWORDS = {
    "stationary": "stationary",
    "abandoned": "abandoned",
    "verified": "verified",
    "removed": "removed",
    "interact": "interacted_with"
}

class QueryInterpreter:
    def __init__(self):
        pass

    def parse(self, query: str) -> dict:
        """
        Parses a natural language string and returns a dictionary of filters.
        Example outputs:
        {"brand": "Dell", "product_type": "Laptop"}
        {"event": "stationary"}
        {"text": "classmate"}
        """
        filters = {}
        q = query.lower()

        # 1. Text Search (e.g. "containing the text CLASSMATE" or "text is CLASSMATE")
        text_match = re.search(r'text\s+([a-z0-9_]+)', q)
        if text_match:
            filters["text"] = text_match.group(1)

        # 2. Brands
        for b_low, b_exact in VALID_BRANDS.items():
            if re.search(r'\b' + re.escape(b_low) + r'\b', q):
                filters["brand"] = b_exact
                break

        # 3. Product Types
        for p_low, p_exact in VALID_PRODUCT_TYPES.items():
            if re.search(r'\b' + re.escape(p_low) + r'\b', q) or re.search(r'\b' + re.escape(p_low) + r's\b', q): # handle plurals crudely
                filters["product_type"] = p_exact
                break

        # 4. Events
        for ev_word, ev_exact in EVENT_KEYWORDS.items():
            if ev_word in q:
                filters["event"] = ev_exact
                break

        # 5. Categories
        for cat in VALID_CATEGORIES:
            if re.search(r'\b' + re.escape(cat) + r'\b', q) or re.search(r'\b' + re.escape(cat) + r's\b', q):
                filters["category"] = cat
                break
                
        # 6. Fallback Text Search
        # If no filters matched, and the query is just a single word or short phrase, treat it as a generic text search.
        if not filters:
            # Clean common prefixes like "show me", "find", "search for"
            clean_q = re.sub(r'^(show\s+me\s+|show\s+|find\s+|search\s+for\s+)', '', q).strip()
            if clean_q and len(clean_q.split()) <= 3:
                filters["text"] = clean_q
                
        # 7. Time ranges - "today" is implied for session.
        # Could parse "between X and Y" but skipping for this simple implementation as requested.

        log.info("Parsed query '%s' -> %s", query, filters)
        return filters
