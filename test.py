import requests
import json

headers = {'Content-Type': 'application/json'}

resp = requests.post(
    "http://localhost:8000/inference",
    headers=headers,
    data=json.dumps({"prompt": "a cat smoking a pipe"})
)

response = resp.json()
print(response)