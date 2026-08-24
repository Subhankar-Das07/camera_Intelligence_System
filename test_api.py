import urllib.request, json
req = urllib.request.Request('http://localhost:8000/api/start_analysis', data=b'{"pipeline_name": "fall_detection_v2"}', headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode())
except urllib.error.HTTPError as e:
    print(e.code, e.read().decode())
except Exception as e:
    print(e)
