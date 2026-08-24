import urllib.request, json
payload = {
    'video_id': 'test',
    'filename': 'test',
    'pipeline_name': 'new_intrusion',
    'roi_normalized': [],
    'config': {},
    'stream_id': None
}
req = urllib.request.Request('http://localhost:8000/api/start_analysis', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode())
except urllib.error.HTTPError as e:
    print(e.code, e.read().decode())
except Exception as e:
    print(e)
