import urllib.request
try:
    with urllib.request.urlopen('http://localhost:8000/api/stream/d05c073c-d649-4dca-95ae-16171369534d', timeout=2) as response:
        pass
except Exception:
    pass
