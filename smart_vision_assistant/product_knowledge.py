"""
product_knowledge.py
====================
Local lightweight Knowledge Base mapping extracted OCR brand text 
to product types. Acts as supporting evidence alongside YOLO detections.
"""

# A dictionary mapping normalised (lowercase) substrings to their metadata.
PRODUCT_KB = {
    # ── Pens & Stationery ─────────────────────────────────────────────
    "cello":      {"brand": "Cello",      "product_type": "Pen", "expected_category": "book"}, 
    "parker":     {"brand": "Parker",     "product_type": "Pen", "expected_category": "book"},
    "reynolds":   {"brand": "Reynolds",   "product_type": "Pen", "expected_category": "book"},
    "linc":       {"brand": "Linc",       "product_type": "Pen", "expected_category": "book"},
    "flair":      {"brand": "Flair",      "product_type": "Pen", "expected_category": "book"},
    "camlin":     {"brand": "Camlin",     "product_type": "Stationery", "expected_category": "book"},
    "apsara":     {"brand": "Apsara",     "product_type": "Pencil/Stationery", "expected_category": "book"},
    "natraj":     {"brand": "Natraj",     "product_type": "Pencil/Stationery", "expected_category": "book"},
    "classmate":  {"brand": "Classmate",  "product_type": "Notebook", "expected_category": "book"},
    "navneet":    {"brand": "Navneet",    "product_type": "Notebook", "expected_category": "book"},
    
    # ── Beverages & Water ─────────────────────────────────────────────
    "bisleri":    {"brand": "Bisleri",    "product_type": "Water Bottle", "expected_category": "bottle"},
    "kinley":     {"brand": "Kinley",     "product_type": "Water Bottle", "expected_category": "bottle"},
    "aquafina":   {"brand": "Aquafina",   "product_type": "Water Bottle", "expected_category": "bottle"},
    "bailley":    {"brand": "Bailley",    "product_type": "Water Bottle", "expected_category": "bottle"},
    "coca cola":  {"brand": "Coca-Cola",  "product_type": "Beverage", "expected_category": "bottle"},
    "cocacola":   {"brand": "Coca-Cola",  "product_type": "Beverage", "expected_category": "bottle"},
    "pepsi":      {"brand": "Pepsi",      "product_type": "Beverage", "expected_category": "bottle"},
    "sprite":     {"brand": "Sprite",     "product_type": "Beverage", "expected_category": "bottle"},
    "thums up":   {"brand": "Thums Up",   "product_type": "Beverage", "expected_category": "bottle"},
    "fanta":      {"brand": "Fanta",      "product_type": "Beverage", "expected_category": "bottle"},
    "mazaa":      {"brand": "Mazaa",      "product_type": "Beverage", "expected_category": "bottle"},
    "slice":      {"brand": "Slice",      "product_type": "Beverage", "expected_category": "bottle"},
    "red bull":   {"brand": "Red Bull",   "product_type": "Energy Drink", "expected_category": "bottle"},
    "gatorade":   {"brand": "Gatorade",   "product_type": "Sports Drink", "expected_category": "bottle"},
    "nescafe":    {"brand": "Nescafe",    "product_type": "Coffee", "expected_category": "cup"},
    "bru":        {"brand": "Bru",        "product_type": "Coffee", "expected_category": "cup"},
    "taj mahal":  {"brand": "Taj Mahal",  "product_type": "Tea", "expected_category": "cup"},
    "red label":  {"brand": "Red Label",  "product_type": "Tea", "expected_category": "cup"},

    # ── Electronics & IT ──────────────────────────────────────────────
    "dell":       {"brand": "Dell",       "product_type": "Laptop/Monitor", "expected_category": "laptop"},
    "hp":         {"brand": "HP",         "product_type": "Laptop", "expected_category": "laptop"},
    "lenovo":     {"brand": "Lenovo",     "product_type": "Laptop", "expected_category": "laptop"},
    "asus":       {"brand": "Asus",       "product_type": "Laptop", "expected_category": "laptop"},
    "acer":       {"brand": "Acer",       "product_type": "Laptop", "expected_category": "laptop"},
    "apple":      {"brand": "Apple",      "product_type": "Device", "expected_category": "laptop"},
    "macbook":    {"brand": "Apple",      "product_type": "MacBook", "expected_category": "laptop"},
    "ipad":       {"brand": "Apple",      "product_type": "iPad", "expected_category": "tv"},
    "iphone":     {"brand": "Apple",      "product_type": "iPhone", "expected_category": "cell phone"},
    "samsung":    {"brand": "Samsung",    "product_type": "Device", "expected_category": "cell phone"},
    "sony":       {"brand": "Sony",       "product_type": "Device", "expected_category": "tv"},
    "lg":         {"brand": "LG",         "product_type": "Device", "expected_category": "tv"},
    "panasonic":  {"brand": "Panasonic",  "product_type": "Device", "expected_category": "tv"},
    "xiaomi":     {"brand": "Xiaomi",     "product_type": "Phone/Device", "expected_category": "cell phone"},
    "redmi":      {"brand": "Redmi",      "product_type": "Phone", "expected_category": "cell phone"},
    "oneplus":    {"brand": "OnePlus",    "product_type": "Phone", "expected_category": "cell phone"},
    "vivo":       {"brand": "Vivo",       "product_type": "Phone", "expected_category": "cell phone"},
    "oppo":       {"brand": "Oppo",       "product_type": "Phone", "expected_category": "cell phone"},
    "realme":     {"brand": "Realme",     "product_type": "Phone", "expected_category": "cell phone"},
    "logitech":   {"brand": "Logitech",   "product_type": "Peripheral", "expected_category": "mouse"},
    "razer":      {"brand": "Razer",      "product_type": "Peripheral", "expected_category": "mouse"},
    "corsair":    {"brand": "Corsair",    "product_type": "Peripheral", "expected_category": "keyboard"},
    
    # ── Food & Snacks ─────────────────────────────────────────────────
    "maggi":      {"brand": "Maggi",      "product_type": "Noodles", "expected_category": "bowl"},
    "yippee":     {"brand": "Yippee",     "product_type": "Noodles", "expected_category": "bowl"},
    "amul":       {"brand": "Amul",       "product_type": "Dairy/Food", "expected_category": "bottle"},
    "britannia":  {"brand": "Britannia",  "product_type": "Biscuits/Food", "expected_category": "sandwich"},
    "parle":      {"brand": "Parle",      "product_type": "Biscuits", "expected_category": "sandwich"},
    "parle g":    {"brand": "Parle-G",    "product_type": "Biscuits", "expected_category": "sandwich"},
    "nestle":     {"brand": "Nestle",     "product_type": "Food Product", "expected_category": "cup"},
    "haldiram":   {"brand": "Haldiram's", "product_type": "Snacks", "expected_category": "bowl"},
    "lays":       {"brand": "Lay's",      "product_type": "Chips", "expected_category": "bowl"},
    "kurkure":    {"brand": "Kurkure",    "product_type": "Snacks", "expected_category": "bowl"},
    "doritos":    {"brand": "Doritos",    "product_type": "Chips", "expected_category": "bowl"},
    "pringles":   {"brand": "Pringles",   "product_type": "Chips", "expected_category": "bottle"},
    "cadbury":    {"brand": "Cadbury",    "product_type": "Chocolate", "expected_category": "sandwich"},
    "dairy milk": {"brand": "Dairy Milk", "product_type": "Chocolate", "expected_category": "sandwich"},
    "kinder":     {"brand": "Kinder",     "product_type": "Chocolate", "expected_category": "sandwich"},
    "snickers":   {"brand": "Snickers",   "product_type": "Chocolate", "expected_category": "sandwich"},
    "kelloggs":   {"brand": "Kellogg's",  "product_type": "Cereal", "expected_category": "bowl"},
    
    # ── Personal Care & Household ─────────────────────────────────────
    "dove":       {"brand": "Dove",       "product_type": "Personal Care", "expected_category": "bottle"},
    "nivea":      {"brand": "Nivea",      "product_type": "Personal Care", "expected_category": "bottle"},
    "colgate":    {"brand": "Colgate",    "product_type": "Toothpaste", "expected_category": "toothbrush"},
    "pepsodent":  {"brand": "Pepsodent",  "product_type": "Toothpaste", "expected_category": "toothbrush"},
    "sensodyne":  {"brand": "Sensodyne",  "product_type": "Toothpaste", "expected_category": "toothbrush"},
    "oral b":     {"brand": "Oral-B",     "product_type": "Toothbrush", "expected_category": "toothbrush"},
    "gillette":   {"brand": "Gillette",   "product_type": "Razor/Grooming", "expected_category": "scissors"},
    "dettol":     {"brand": "Dettol",     "product_type": "Antiseptic/Soap", "expected_category": "bottle"},
    "savlon":     {"brand": "Savlon",     "product_type": "Antiseptic/Soap", "expected_category": "bottle"},
    "lifebuoy":   {"brand": "Lifebuoy",   "product_type": "Soap", "expected_category": "bottle"},
    "lux":        {"brand": "Lux",        "product_type": "Soap", "expected_category": "bottle"},
    "pears":      {"brand": "Pears",      "product_type": "Soap", "expected_category": "bottle"},
    "sunsilk":    {"brand": "Sunsilk",    "product_type": "Shampoo", "expected_category": "bottle"},
    "clinic plus":{"brand": "Clinic Plus","product_type": "Shampoo", "expected_category": "bottle"},
    "head shoulders":{"brand": "Head & Shoulders","product_type": "Shampoo", "expected_category": "bottle"},
    "pantene":    {"brand": "Pantene",    "product_type": "Shampoo", "expected_category": "bottle"},
    "tresemme":   {"brand": "TRESemme",   "product_type": "Shampoo", "expected_category": "bottle"},
    "himalaya":   {"brand": "Himalaya",   "product_type": "Personal Care", "expected_category": "bottle"},
    "patanjali":  {"brand": "Patanjali",  "product_type": "FMCG", "expected_category": "bottle"},
    "surf excel": {"brand": "Surf Excel", "product_type": "Detergent", "expected_category": "bottle"},
    "ariel":      {"brand": "Ariel",      "product_type": "Detergent", "expected_category": "bottle"},
    "tide":       {"brand": "Tide",       "product_type": "Detergent", "expected_category": "bottle"},
    "vim":        {"brand": "Vim",        "product_type": "Dishwash", "expected_category": "bottle"},
    "lizol":      {"brand": "Lizol",      "product_type": "Disinfectant", "expected_category": "bottle"},
    "harpic":     {"brand": "Harpic",     "product_type": "Cleaner", "expected_category": "bottle"},
    "colin":      {"brand": "Colin",      "product_type": "Glass Cleaner", "expected_category": "bottle"},
    "hit":        {"brand": "HIT",        "product_type": "Insect Repellent", "expected_category": "bottle"},
    "all out":    {"brand": "All Out",    "product_type": "Insect Repellent", "expected_category": "bottle"},
    "good knight":{"brand": "Good Knight","product_type": "Insect Repellent", "expected_category": "bottle"}
}

def lookup_brand(text: str) -> dict:
    """
    Looks for brand matches in the provided text.
    Case-insensitive substring match. Returns dict or empty dict.
    """
    if not text:
        return {}
        
    text_lower = text.strip().lower()
    
    # We test for substrings, preferring exact or distinct word matches if possible.
    # We sort by length descending to match 'coca cola' before 'coca'
    sorted_keys = sorted(PRODUCT_KB.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        if key in text_lower:
            return PRODUCT_KB[key]
            
    return {}
