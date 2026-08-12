"""
entity_inference.py
===================
The Evidence Fusion Layer.
Synthesizes visual signals (YOLO class) and text signals (OCR text / Brand KB) 
into a higher-confidence inferred label, without overwriting raw underlying data.
"""

def infer_entity_identity(yolo_label: str, ocr_text: str, kb_match: dict) -> str:
    """
    Generate a human-readable display label by combining evidence.
    Does not replace YOLO classes, but enhances them.
    
    Returns inferred string, or just yolo_label if no extra evidence exists.
    """
    if not ocr_text:
        return yolo_label
        
    if kb_match:
        brand = kb_match["brand"]
        product_type = kb_match["product_type"]
        expected_cat = kb_match.get("expected_category", "")
        
        # Scenario 1: YOLO and KB agree on the category (e.g. YOLO=bottle, KB=Water Bottle)
        if yolo_label == expected_cat or expected_cat in yolo_label:
            return f"{brand} {product_type}"
            
        # Scenario 2: YOLO missed the class but we have strong OCR evidence 
        # (or the YOLO class is something generic like 'unknown' or 'object')
        if yolo_label in ["unknown", "object"]:
            return f"Likely {brand} {product_type}"
            
        # Scenario 3: YOLO says 'person' but KB matched 'Apple' (maybe a t-shirt logo).
        # We don't want to call the person an 'Apple Device'.
        if yolo_label == "person":
            return f"{yolo_label} (wearing {brand})"
            
        # Scenario 4: Conflicting evidence. We prepend the brand to the YOLO class.
        # e.g., YOLO says 'cup', KB says 'Dell Laptop'. We output "Dell cup" (or similar)
        # to show the text was read on that object.
        return f"{brand} {yolo_label}"
        
    else:
        # We have text but no KB match. Just append it for context.
        # e.g., "book (CLASSMATE)"
        if yolo_label == "person":
            return f"{yolo_label} [{ocr_text}]"
        return f"{yolo_label} ({ocr_text})"
