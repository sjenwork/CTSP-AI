import requests
from pprint import pprint

data_url = "https://raw.githubusercontent.com/sjenwork/CTSP-AI/main/dev/ctsp/example_input.json"
res = requests.get(data_url)
if res.ok:
    data = res.json()

post_url = "http://127.0.0.1:8000/predict"

response = requests.post(post_url, json=data)
print(response.json())
